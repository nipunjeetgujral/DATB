import os, io, numpy as np, pandas as pd, torch
from torch import nn
from sqlalchemy import create_engine

engine = create_engine(os.environ.get('PG_DSN'))
FEATURES = ['ret_5m','vol_1h','rsi_14','ema_12','ema_48','bb_upper','bb_lower']
H = 30*24*12

class Encoder(nn.Module):
    def __init__(self, nfeat, hid=64):
        super().__init__(); self.lstm = nn.LSTM(nfeat, hid, 2, batch_first=True)
    def forward(self, x): out,(h,c)=self.lstm(x); return (h,c)

class Decoder(nn.Module):
    def __init__(self, hid=64):
        super().__init__(); self.lstm = nn.LSTM(1, hid, 2, batch_first=True); self.fc=nn.Linear(hid,1)
    def forward(self, Y, hc):
        h,c=hc; out,_=self.lstm(Y,(h,c)); return self.fc(out)

def _load_train_df():
    q = f"""
    SELECT f.ts, {','.join([f'f.{c}' for c in FEATURES])}, p.close
    FROM btc_features_5m f JOIN btc_5m p ON p.ts=f.ts
    WHERE f.ts > now() - interval '3 years'
    ORDER BY f.ts
    """
    return pd.read_sql(q, engine).dropna()

def _make_windows(df, lookback=288, horizon=H, stride=12):
    Xs, Ys = [], []
    F = df[FEATURES].values.astype('float32')
    P = df['close'].values.astype('float32')
    for i in range(lookback, len(df)-horizon, stride):
        Xs.append(F[i-lookback:i])
        Ys.append(P[i:i+horizon].reshape(-1,1))
    return np.array(Xs), np.array(Ys)

def train_lstm_seq2seq_3y(**_):
    df = _load_train_df()
    if len(df) < (288 + H + 100): return
    X, Y = _make_windows(df)
    enc, dec = Encoder(len(FEATURES),64), Decoder(64)
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-3)
    loss = nn.L1Loss()
    enc.train(); dec.train()
    for epoch in range(5):
        idx = np.random.permutation(len(X))
        for j in range(0, len(X), 16):
            xb = torch.from_numpy(X[idx[j:j+16]])
            yb = torch.from_numpy(Y[idx[j:j+16]])
            hc = enc(xb)
            pred = dec(yb[:,:-1,:], hc)  # teacher forcing
            l = loss(pred, yb)
            opt.zero_grad(); l.backward(); opt.step()
    blob = io.BytesIO()
    torch.save({'enc':enc.state_dict(),'dec':dec.state_dict()}, blob); blob.seek(0)
    meta = {'features':FEATURES, 'lookback':288, 'horizon':H, 'step':5}
    v = pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')
    with engine.begin() as conn:
        conn.execute("""
          CREATE TABLE IF NOT EXISTS model_registry (model_name text, version text, created_at timestamptz DEFAULT now(),
            binary bytea, meta jsonb, PRIMARY KEY (model_name, version))
        """)
        conn.execute("INSERT INTO model_registry (model_name,version,binary,meta) VALUES (%s,%s,%s,%s)",
                     ('lstm_seq2seq', v, blob.getvalue(), meta))
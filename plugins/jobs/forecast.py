import os, io, numpy as np, pandas as pd, torch
from torch import nn
from sqlalchemy import create_engine

engine = create_engine(os.environ.get('PG_DSN'))
FEATURES = ['ret_5m','vol_1h','rsi_14','ema_12','ema_48','bb_upper','bb_lower']
H = 30*24*12   # 8640 steps
STEP = 5

class Encoder(nn.Module):
    def __init__(self, nfeat, hid=64):
        super().__init__(); self.lstm = nn.LSTM(nfeat, hid, 2, batch_first=True)
    def forward(self, x): out,(h,c)=self.lstm(x); return (h,c)

class Decoder(nn.Module):
    def __init__(self, hid=64):
        super().__init__(); self.lstm = nn.LSTM(1, hid, 2, batch_first=True); self.fc=nn.Linear(hid,1)
    def forward(self, y0, hc):
        y=y0; outs=[]; h,c=hc
        for _ in range(H):
            o,(h,c)=self.lstm(y,(h,c))
            y=self.fc(o[:,-1:,:]); outs.append(y)
        return torch.cat(outs, dim=1)

def _load_latest():
    row = engine.execute("SELECT version,binary,meta FROM model_registry WHERE model_name='lstm_seq2seq' ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row: return None
    ver, blob, meta = row
    state = torch.load(io.BytesIO(bytes(blob)), map_location='cpu')
    enc, dec = Encoder(len(FEATURES),64), Decoder(64)
    enc.load_state_dict(state['enc']); dec.load_state_dict(state['dec'])
    enc.eval(); dec.eval()
    return enc, dec, ver

def _latest_window(L=288):
    q = f"""
      SELECT f.ts, {','.join([f'f.{c}' for c in FEATURES])}, p.close
      FROM btc_features_5m f JOIN btc_5m p ON p.ts=f.ts
      ORDER BY f.ts DESC LIMIT {L+3}
    """
    df = pd.read_sql(q, engine).sort_values('ts').tail(L)
    X = df[FEATURES].values.astype('float32')
    last_close = float(df['close'].iloc[-1])
    ts_in = df['ts'].iloc[-1]
    return ts_in, X, last_close

@torch.no_grad()
def run_inference_path(**_):
    mdl = _load_latest()
    if mdl is None: return
    enc, dec, ver = mdl
    ts_in, X, last_close = _latest_window()
    hc = enc(torch.from_numpy(np.expand_dims(X,0)))
    y0 = torch.tensor([[[last_close]]], dtype=torch.float32)
    path = dec(y0, hc).cpu().numpy().ravel().tolist()
    with engine.begin() as conn:
        conn.execute("""
          CREATE TABLE IF NOT EXISTS btc_forecast_1mo_path (
            ts_input timestamptz PRIMARY KEY, horizon_minutes int, step_minutes int,
            yhat_path double precision[], model_version text
          )""")
        conn.execute("""
          INSERT INTO btc_forecast_1mo_path (ts_input, horizon_minutes, step_minutes, yhat_path, model_version)
          VALUES (%s,%s,%s,%s,%s)
          ON CONFLICT (ts_input) DO UPDATE SET yhat_path=EXCLUDED.yhat_path, model_version=EXCLUDED.model_version
        """, (ts_in.to_pydatetime(), 30*24*60, 5, path, ver))
import os, io, json, numpy as np, pandas as pd, torch
from torch import nn
from sqlalchemy import create_engine
from .rl import DQN, ACTIONS

engine = create_engine(os.environ.get('PG_DSN'))

def _state_at_ts(ts):
    row = engine.execute("""
      WITH f AS (
        SELECT yhat_path[1] AS first, yhat_path[array_length(yhat_path,1)] AS last
        FROM btc_forecast_1mo_path WHERE ts_input <= %s ORDER BY ts_input DESC LIMIT 1),
      p AS (SELECT close FROM btc_5m WHERE ts <= %s ORDER BY ts DESC LIMIT 1),
      s AS (SELECT comp_30m, comp_2h, comp_1d FROM btc_sentiment_rollups WHERE ts_bucket <= %s ORDER BY ts_bucket DESC LIMIT 1),
      pos AS (SELECT btc_position FROM portfolio WHERE ts <= %s ORDER BY ts DESC LIMIT 1)
      SELECT p.close, f.first, f.last, s.comp_30m, s.comp_2h, s.comp_1d, coalesce(pos.btc_position,0)
      FROM f,p,s LEFT JOIN pos ON true
    """, (ts,ts,ts,ts)).fetchone()
    if not row: return None
    price, first, last, c30, c2h, c1d, position = row
    edge_short = (first - price)/max(1.0, price)
    edge_long  = (last  - price)/max(1.0, price)
    return np.array([edge_short, edge_long, c30 or 0, c2h or 0, c1d or 0, position or 0], dtype=np.float32)

def _build_replay():
    df = pd.read_sql("""
      SELECT a.ts, a.action, a.price, p.btc_nav
      FROM rl_actions a JOIN portfolio p ON p.ts=a.ts
      WHERE a.ts > now() - interval '90 days' ORDER BY a.ts
    """, engine)
    if df.empty: return None
    S,A,R,S2=[],[],[],[]
    prev_nav=None; prev_s=None
    for _, row in df.iterrows():
        ts=row.ts
        s = _state_at_ts(ts)
        if s is None: continue
        if prev_s is not None and prev_nav is not None:
            R.append(row.btc_nav - prev_nav)
            S.append(prev_s); S2.append(s)
        A.append(ACTIONS.index(row.action))
        prev_nav=row.btc_nav; prev_s=s
    if not S: return None
    return np.stack(S), np.array(A[:len(S)]), np.array(R), np.stack(S2)

def train_dqn_daily(**_):
    buf = _build_replay()
    if buf is None: return
    S, A, R, S2 = buf
    state_dim = S.shape[1]
    policy, target = DQN(state_dim), DQN(state_dim)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss = nn.SmoothL1Loss(); gamma=0.99
    for _ in range(200):
        idx = np.random.randint(0, len(S), size=min(64, len(S)))
        s  = torch.tensor(S[idx]); a = torch.tensor(A[idx]).long().view(-1,1)
        r  = torch.tensor(R[idx]).float().view(-1,1)
        sp = torch.tensor(S2[idx])
        q  = policy(s).gather(1, a)
        with torch.no_grad():
            qn = target(sp).max(1, keepdim=True)[0]
        l = loss(q, r + gamma*qn)
        opt.zero_grad(); l.backward(); opt.step()
    meta={'state_dim': state_dim, 'actions': ACTIONS}
    b = io.BytesIO(); torch.save(policy.state_dict(), b); b.seek(0)
    v = pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')
    with engine.begin() as conn:
        conn.execute("""
          CREATE TABLE IF NOT EXISTS model_registry (model_name text, version text, created_at timestamptz DEFAULT now(),
            binary bytea, meta jsonb, PRIMARY KEY (model_name, version))
        """ )
        conn.execute('INSERT INTO model_registry (model_name,version,binary,meta) VALUES (%s,%s,%s,%s)',
                     ('dqn_policy', v, b.getvalue(), json.dumps(meta)))
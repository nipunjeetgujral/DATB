import os, io, json, numpy as np, pandas as pd, torch
from torch import nn
from sqlalchemy import create_engine

engine = create_engine(os.environ.get('PG_DSN'))
ACTIONS = ['BUY','SELL','HOLD']; NOTIONAL = 100.0

class DQN(nn.Module):
    def __init__(self, n):
        super().__init__(); self.net = nn.Sequential(
            nn.Linear(n,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,3))
    def forward(self,x): return self.net(x)

def _latest_state():
    row = engine.execute("""
      WITH f AS (SELECT ts_input, yhat_path[1] AS first, yhat_path[array_length(yhat_path,1)] AS last
                 FROM btc_forecast_1mo_path ORDER BY ts_input DESC LIMIT 1),
           p AS (SELECT ts, close FROM btc_5m ORDER BY ts DESC LIMIT 1),
           s AS (SELECT comp_30m, comp_2h, comp_1d FROM btc_sentiment_rollups ORDER BY ts_bucket DESC LIMIT 1)
      SELECT f.ts_input, p.close, f.first, f.last, s.comp_30m, s.comp_2h, s.comp_1d
      FROM f, p, s
    """).fetchone()
    if not row: return None
    ts, price, first, last, c30, c2h, c1d = row
    edge_short = (first - price)/max(1.0, price)
    edge_long  = (last  - price)/max(1.0, price)
    pos = engine.execute('SELECT btc_position FROM portfolio ORDER BY ts DESC LIMIT 1').scalar() or 0.0
    s = np.array([edge_short, edge_long, c30 or 0, c2h or 0, c1d or 0, pos], dtype=np.float32)
    return ts, float(price), s

def _load_policy():
    row = engine.execute("SELECT version,binary,meta FROM model_registry WHERE model_name='dqn_policy' ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row: return None, None
    v, blob, meta = row; meta=dict(meta)
    m = DQN(meta['state_dim']); m.load_state_dict(torch.load(io.BytesIO(bytes(blob)), map_location='cpu')); m.eval()
    return m, meta

def rl_decide_and_execute(**_):
    policy, meta = _load_policy()
    state = _latest_state()
    if state is None: return
    ts, price, s = state
    if policy is None:
        a = 2   # HOLD
    else:
        with torch.no_grad():
            a = int(torch.argmax(policy(torch.tensor(s).unsqueeze(0)), dim=1))
    action = ['BUY','SELL','HOLD'][a]
    qty = NOTIONAL/price if action=='BUY' else (-NOTIONAL/price if action=='SELL' else 0.0)
    with engine.begin() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS rl_actions (
            ts timestamptz PRIMARY KEY, action text, notional_usd double precision, price double precision, qty_btc double precision)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS portfolio (
            ts timestamptz PRIMARY KEY, btc_position double precision, usd_cash double precision, btc_nav double precision)""")
        if qty != 0:
            conn.execute("INSERT INTO rl_actions (ts, action, notional_usd, price, qty_btc) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (ts) DO NOTHING",
                         (ts, action, NOTIONAL, price, qty))
            last = conn.execute('SELECT btc_position, usd_cash FROM portfolio ORDER BY ts DESC LIMIT 1').fetchone() or (0.0,0.0)
            pos = last[0] + qty; cash = last[1] - qty*price
            nav = pos + cash/price
            conn.execute('INSERT INTO portfolio (ts, btc_position, usd_cash, btc_nav) VALUES (%s,%s,%s,%s) ON CONFLICT (ts) DO NOTHING',
                         (ts, pos, cash, nav))
        else:
            last = conn.execute('SELECT btc_position, usd_cash FROM portfolio ORDER BY ts DESC LIMIT 1').fetchone() or (0.0,0.0)
            nav = last[0] + (last[1]/price)
            conn.execute('INSERT INTO portfolio (ts, btc_position, usd_cash, btc_nav) VALUES (%s,%s,%s,%s) ON CONFLICT (ts) DO NOTHING',
                         (ts, last[0], last[1], nav))
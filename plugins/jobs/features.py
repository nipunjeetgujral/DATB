import os, pandas as pd, numpy as np
from sqlalchemy import create_engine

engine = create_engine(os.environ.get('PG_DSN'))

def update_features_5m(**_):
    df = pd.read_sql('SELECT * FROM btc_5m ORDER BY ts', engine)
    if df.empty: return
    df['ret_5m'] = np.log(df['close']).diff()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_48'] = df['close'].ewm(span=48, adjust=False).mean()
    df['vol_1h'] = df['ret_5m'].rolling(12).std()
    delta = df['close'].diff()
    up = delta.clip(lower=0); down=(-delta).clip(lower=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean()/(down.ewm(alpha=1/14, adjust=False).mean()+1e-9)
    df['rsi_14'] = 100 - (100/(1+rs))
    m = df['close'].rolling(20).mean(); s = df['close'].rolling(20).std()
    df['bb_upper'] = m + 2*s; df['bb_lower'] = m - 2*s
    out = df.dropna().copy()
    with engine.begin() as conn:
        out[['ts','ret_5m','vol_1h','rsi_14','ema_12','ema_48','bb_upper','bb_lower']].to_sql(
            'btc_features_5m', conn, if_exists='append', index=False, method='multi')
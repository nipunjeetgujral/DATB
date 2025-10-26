import os, requests, datetime as dt, psycopg2
from psycopg2.extras import execute_values

PG_DSN = os.environ.get('PG_DSN')
TOKEN  = os.environ.get('TIINGO_TOKEN')
S = requests.Session(); S.headers['Authorization'] = f'Token {TOKEN}'

def _ts(s):
    return dt.datetime.fromisoformat(s.replace('Z','+00:00'))

def collect_prices_5m(**_):
    start = _get_last_ts() or (dt.datetime.utcnow()-dt.timedelta(days=5))
    url = 'https://api.tiingo.com/tiingo/crypto/prices'
    params = {'tickers': 'btcusd', 'resampleFreq': '5min',
              'startDate': start.isoformat(timespec='seconds')+'Z', 'format': 'json'}
    r = S.get(url, params=params, timeout=30); r.raise_for_status()
    data = r.json()
    rows = []
    for d in data[0]['priceData']:
        rows.append((_ts(d['date']), d['open'], d['high'], d['low'], d['close'], d.get('volume',0)))
    if rows: _upsert('btc_5m', rows)

def collect_news(**_):
    start = _get_last_news_ts() or (dt.datetime.utcnow()-dt.timedelta(days=14))
    url = 'https://api.tiingo.com/tiingo/news'
    r = S.get(url, params={'tickers':'BTC','startDate':start.isoformat()+'Z','limit':100}, timeout=30); r.raise_for_status()
    data = r.json()
    rows = []
    for a in data:
        rows.append((str(a['id']), _ts(a['publishedDate']), a.get('title',''), a.get('url',''),
                     a.get('source','tiingo'), a.get('description','')))
    if rows: _upsert_articles(rows)

def _get_last_ts():
    with psycopg2.connect(PG_DSN) as c, c.cursor() as cur:
        cur.execute('CREATE TABLE IF NOT EXISTS btc_5m (ts timestamptz PRIMARY KEY, open double precision, high double precision, low double precision, close double precision, volume double precision, source text DEFAULT ''tiingo'')')
        cur.execute('SELECT max(ts) FROM btc_5m'); return cur.fetchone()[0]

def _get_last_news_ts():
    with psycopg2.connect(PG_DSN) as c, c.cursor() as cur:
        cur.execute('CREATE TABLE IF NOT EXISTS btc_articles (id text PRIMARY KEY, published_at timestamptz, title text, url text, source text, body text)')
        cur.execute('SELECT max(published_at) FROM btc_articles'); return cur.fetchone()[0]

def _upsert(table, rows):
    with psycopg2.connect(PG_DSN) as c, c.cursor() as cur:
        execute_values(cur, f"""
          INSERT INTO {table} (ts, open, high, low, close, volume)
          VALUES %s
          ON CONFLICT (ts) DO UPDATE SET
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume
        """, rows)

def _upsert_articles(rows):
    with psycopg2.connect(PG_DSN) as c, c.cursor() as cur:
        execute_values(cur, """
          INSERT INTO btc_articles (id, published_at, title, url, source, body)
          VALUES %s ON CONFLICT (id) DO NOTHING
        """, rows)
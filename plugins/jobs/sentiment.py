import os, pandas as pd
from sqlalchemy import create_engine
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk; nltk.download('vader_lexicon', quiet=True)

engine = create_engine(os.environ.get('PG_DSN'))
sia = SentimentIntensityAnalyzer()

def score_new_articles(**_):
    q = """SELECT a.id, a.title||' '||coalesce(a.body,'') AS text
           FROM btc_articles a LEFT JOIN btc_sentiment s ON s.article_id=a.id
           WHERE s.article_id IS NULL ORDER BY a.published_at"""
    df = pd.read_sql(q, engine)
    if df.empty: return
    df[['neg','neu','pos','compound']] = df['text'].apply(lambda t: pd.Series(sia.polarity_scores(t)))
    with engine.begin() as conn:
        df[['id','compound','pos','neu','neg']].rename(columns={'id':'article_id'}).to_sql(
            'btc_sentiment', conn, if_exists='append', index=False, method='multi')

def rollup_sentiment(**_):
    q = """
      SELECT date_trunc('minute', published_at) ts, compound
      FROM btc_articles a JOIN btc_sentiment s ON s.article_id=a.id
      WHERE published_at > now()-interval '3 days'
    """
    df = pd.read_sql(q, engine)
    if df.empty: return
    g = df.groupby('ts')['compound'].mean()
    out = pd.DataFrame({
        'ts_bucket': g.index,
        'comp_30m': g.rolling(30, min_periods=1).mean().values,
        'comp_2h':  g.rolling(24, min_periods=1).mean().values,
        'comp_1d':  g.rolling(288, min_periods=1).mean().values
    })
    with engine.begin() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS btc_sentiment_rollups (ts_bucket timestamptz PRIMARY KEY, comp_30m double precision, comp_2h double precision, comp_1d double precision)")
        conn.execute("DELETE FROM btc_sentiment_rollups WHERE ts_bucket>=now()-interval '3 days'")
        out.to_sql('btc_sentiment_rollups', conn, if_exists='append', index=False, method='multi')
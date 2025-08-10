import feedparser
import pandas as pd

FEEDS = [
    "https://feeds.feedburner.com/ndtvnews-india-news",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://indianexpress.com/section/india/feed/",
    # Add more feeds you trust
]
OUT = "data/new_real_rss.csv"
rows = []

print("Fetching articles from RSS feeds...")

for url in FEEDS:
    feed = feedparser.parse(url)
    for entry in feed.entries:
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        content = " ".join([title, summary])
        link = getattr(entry, "link", "") or ""
        if len(content) < 20:
            continue
        rows.append({
            "title": title,
            "content": content,
            "url": link,
            "label": 0
        })

df = pd.DataFrame(rows).drop_duplicates(subset=["content"])
df.to_csv(OUT, index=False, encoding="utf-8")
print(f"Saved {len(df)} RSS articles to {OUT}")

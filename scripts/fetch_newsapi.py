import os
import requests
import pandas as pd
import feedparser
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_KEY = os.getenv("GNEWS_KEY")  # get from https://gnews.io/
OUT = "data/aggregated_indian_news.csv"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

INDIAN_SOURCES_NEWSAPI = "the-times-of-india,google-news-in,business-insider-uk,hindustan-times,ndtv"
INDIAN_RSS_FEEDS = [
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://feeds.feedburner.com/ndtvnews-india-news",
]

CATEGORIES = ["business", "entertainment", "health", "science", "sports", "technology"]

def fetch_newsapi_by_sources(sources, page_size=100):
    if not NEWSAPI_KEY:
        print("NEWSAPI_KEY missing, skipping NewsAPI sources fetch")
        return []
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWSAPI_KEY,
        "pageSize": page_size,
        "sources": sources,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        print(f"NewsAPI (sources): fetched {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"NewsAPI error (sources): {e}")
        return []

def fetch_newsapi_by_category(category, page_size=100):
    if not NEWSAPI_KEY:
        print("NEWSAPI_KEY missing, skipping NewsAPI category fetch")
        return []
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWSAPI_KEY,
        "pageSize": page_size,
        "category": category,
        # "country": "in",  # uncomment if you want country but beware free tier often returns zero
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        print(f"NewsAPI (category={category}): fetched {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"NewsAPI error (category={category}): {e}")
        return []

def fetch_gnews(query="India", max_results=100):
    if not GNEWS_KEY:
        print("GNEWS_KEY missing, skipping GNews")
        return []
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "token": GNEWS_KEY,
        "lang": "en",
        "max": max_results,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        print(f"GNews: fetched {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"GNews error: {e}")
        return []

def fetch_rss_feeds(feed_urls):
    articles = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                title = entry.get("title", "")
                desc = entry.get("summary", "")
                link = entry.get("link", "")
                combined = f"{title} {desc}".strip()
                if len(combined) < 10:
                    continue
                articles.append({
                    "title": title,
                    "content": combined,
                    "source": feed.feed.get("title", "RSS Feed"),
                    "url": link,
                    "label": 0,
                })
                count += 1
            print(f"RSS: fetched {count} articles from {url}")
        except Exception as e:
            print(f"RSS fetch error for {url}: {e}")
    return articles

def unify_newsapi_articles(newsapi_articles):
    unified = []
    for a in newsapi_articles:
        title = str(a.get("title") or "")
        desc = str(a.get("description") or "")
        content = str(a.get("content") or "")
        url = str(a.get("url") or "")
        source = str(a.get("source", {}).get("name") or "")
        combined = " ".join([title, desc, content]).strip()
        if len(combined) < 10:
            continue
        unified.append({
            "title": title,
            "content": combined,
            "source": source,
            "url": url,
            "label": 0,
        })
    return unified


def unify_gnews_articles(gnews_articles):
    unified = []
    for a in gnews_articles:
        title = str(a.get("title") or "")
        desc = str(a.get("description") or "")
        content = str(a.get("content") or "")
        url = str(a.get("url") or "")
        source = ""
        if "source" in a and a["source"]:
            source = str(a["source"].get("name") or "")
        combined = " ".join([title, desc, content]).strip()
        if len(combined) < 10:
            continue
        unified.append({
            "title": title,
            "content": combined,
            "source": source,
            "url": url,
            "label": 0,
        })
    return unified


def deduplicate_articles(articles):
    unique = {}
    for art in articles:
        url = art.get("url")
        if url and url not in unique:
            unique[url] = art
    return list(unique.values())

def main():
    all_articles = []

    # 1) Fetch all Indian source news (no category)
    print("Fetching all news from Indian sources (no category)...")
    newsapi_source_articles = fetch_newsapi_by_sources(INDIAN_SOURCES_NEWSAPI)
    all_articles.extend(unify_newsapi_articles(newsapi_source_articles))

    # 2) Fetch category-wise global news (no sources param)
    for cat in CATEGORIES:
        print(f"\nFetching global category '{cat}' news (no sources)...")
        cat_articles = fetch_newsapi_by_category(cat)
        all_articles.extend(unify_newsapi_articles(cat_articles))

    # 3) Fetch from GNews API with query "India"
    print("\nFetching news from GNews API with query 'India'...")
    gnews_articles = fetch_gnews(query="India", max_results=50)
    all_articles.extend(unify_gnews_articles(gnews_articles))

    # 4) Fetch articles from Indian RSS feeds
    print("\nFetching news from Indian RSS feeds...")
    rss_articles = fetch_rss_feeds(INDIAN_RSS_FEEDS)
    all_articles.extend(rss_articles)

    print(f"\nTotal articles before deduplication: {len(all_articles)}")
    unique_articles = deduplicate_articles(all_articles)
    print(f"Total unique articles after deduplication: {len(unique_articles)}")

    if unique_articles:
        df = pd.DataFrame(unique_articles)
        df.to_csv(OUT, index=False, encoding="utf-8")
        print(f"Saved {len(unique_articles)} articles to {OUT}")
    else:
        print("No articles fetched from any source.")

if __name__ == "__main__":
    main()

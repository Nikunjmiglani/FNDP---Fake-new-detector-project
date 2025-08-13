import sys
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
from datetime import datetime

# Force UTF-8 output so emojis don't crash on Windows
sys.stdout.reconfigure(encoding='utf-8')

# List of fake/satire news URLs
FAKE_NEWS_URLS = [
    "https://www.theonion.com/",
    "https://babylonbee.com/",
    "https://www.worldnewsdailyreport.com/",
    "https://waterfordwhispersnews.com/"
]

# Headers to bypass 401 / bot blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/139.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

OUTPUT_FILE = "fake_news.csv"

def scrape_with_newspaper(url):
    try:
        article = Article(url)
        article.download(input_html=requests.get(url, headers=HEADERS, timeout=10).text)
        article.parse()
        return {
            "title": article.title.strip(),
            "text": article.text.strip(),
            "url": url,
            "published": article.publish_date if article.publish_date else datetime.now()
        }
    except Exception as e:
        print(f"❌ [newspaper3k] Failed for {url}: {e}")
        return None

def scrape_with_bs4(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title else "No Title Found"
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text = "\n".join(paragraphs)
        return {
            "title": title,
            "text": text if text else "No content extracted",
            "url": url,
            "published": datetime.now()
        }
    except Exception as e:
        print(f"❌ [BeautifulSoup] Failed for {url}: {e}")
        return None

def scrape_fake_news():
    articles = []
    for url in FAKE_NEWS_URLS:
        print(f"🔍 Scraping: {url}")
        article_data = scrape_with_newspaper(url)
        if not article_data or not article_data["text"]:
            print(f"⚠ Falling back to BeautifulSoup for {url}")
            article_data = scrape_with_bs4(url)

        if article_data and article_data["text"]:
            articles.append(article_data)
        else:
            print(f"⚠ No content found for {url}")

    return articles

if __name__ == "__main__":
    articles = scrape_fake_news()
    df = pd.DataFrame(articles)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} articles to {OUTPUT_FILE}")

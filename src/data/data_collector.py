from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def scrape_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """Scrape content from a list of URLs"""
    documents = []

    for url in tqdm(urls, desc="Scraping URLs"):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title = soup.title.text if soup.title else "No Title"

            # Extract main content (simple approach)
            article_text = ""
            main_content = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])

            for element in main_content:
                article_text += element.text.strip() + "\n\n"

            if article_text.strip():  # Only add if we have content
                documents.append(
                    {
                        "source": url,
                        "title": title,
                        "content": article_text,
                        "metadata": {
                            "url": url,
                            "scrape_date": pd.Timestamp.now().isoformat(),
                        },
                    }
                )
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")

    return documents

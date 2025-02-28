# data_collection.py
import os
import requests

import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent import futures
from typing import List, Dict, Any, Optional



class DataCollector:
    """Abstract base class for data collection. \n
    Data collection class inherit form this.
    """
    def collect(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement collect method")

class WebScraper(DataCollector):
    """Web scraper for collecting text data from websites"""
    
    def __init__(self, urls: List[str], max_workers: int = 5):
        self.urls = urls
        self.max_workers = max_workers
    
    def _scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.title.text if soup.title else "No Title"
            
            # Extract main content (this is a simple approach, you might need to customize per site)
            article_text = ""
            main_content = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            for element in main_content:
                article_text += element.text.strip() + "\n\n"
            
            return {
                "source": url,
                "title": title,
                "content": article_text,
                "metadata": {
                    "url": url,
                    "scrape_date": pd.Timestamp.now().isoformat()
                }
            }
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {
                "source": url,
                "title": "Error",
                "content": "",
                "metadata": {
                    "url": url,
                    "error": str(e)
                }
            }
    
    def collect(self) -> List[Dict[str, Any]]:
        """Scrape URLs in parallel"""
        results = []
        
        with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self._scrape_url, url): url for url in self.urls}
            
            for future in tqdm(futures.as_completed(future_to_url), total=len(self.urls), desc="Scraping URLs"):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result["content"]:  # Only keep non-empty results
                        results.append(result)
                except Exception as e:
                    print(f"Error processing result from {url}: {str(e)}")
        
        return results

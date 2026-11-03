import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scraper.clinicaltrials import fetch_clinicaltrials_data
from scraper.euctr import fetch_euctr_data
from scraper.pubmed import fetch_pubmed_articles
from utils.tokenize import graceful_json_parse, safe_json_dump
import json
import os

def scraper_tool(state: dict) -> dict:
    """
    Scrape clinical trials, EUCTR, and PubMed data for a given target.

    Expected input:
        state = {
            "target": "<search term or identifier>"
        }

    Returns:
        {
            ...state,
            "clinical_trials_scraped_data": <clinical trials data>,
            "euctr_scraped_data": <EUCTR data>,
            "pubmed_scraped_data": <PubMed articles>
        }
    """
    print("---running scrapper tool-------")
    
    try:
        target = state.get("target")
        if not target:
            print("Warning: Missing target in state, cannot proceed with scraping")
            return {
                **state,
                "clinical_trials_scraped_data": [],
                "euctr_scraped_data": [],
                "pubmed_scraped_data": [],
                "message": "Missing target for scraping"
            }
        
        clinical_trials_scraped_data = []
        euctr_scraped_data = []
        pubmed_scraped_data = []
        
        try:
            clinical_trials_scraped_data = fetch_clinicaltrials_data(target=target)
            print("clinical data scraped")
        except Exception as e:
            print(f"Error scraping clinical trials data: {str(e)}")
            clinical_trials_scraped_data = []
        
        try:
            euctr_scraped_data = fetch_euctr_data(target)
            print("ecutr data scraped")
        except Exception as e:
            print(f"Error scraping EUCTR data: {str(e)}")
            euctr_scraped_data = []
        
        try:
            pubmed_scraped_data = fetch_pubmed_articles(target)
            print("pubmed data scraped")
        except Exception as e:
            print(f"Error scraping PubMed data: {str(e)}")
            pubmed_scraped_data = []
        
        print("data scrapped...")
        
        return {
            **state,
            "clinical_trials_scraped_data": clinical_trials_scraped_data,
            "euctr_scraped_data": euctr_scraped_data,
            "pubmed_scraped_data": pubmed_scraped_data
        }
    
    except Exception as e:
        print(f"Critical error in scraper tool: {str(e)}")
        return {
            **state,
            "clinical_trials_scraped_data": [],
            "euctr_scraped_data": [],
            "pubmed_scraped_data": [],
            "message": f"Critical error in scraper tool: {str(e)}",
            "error": str(e)
        }

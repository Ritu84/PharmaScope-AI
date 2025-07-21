import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scraper.clinicaltrials import fetch_clinicaltrials_data
from scraper.euctr import fetch_euctr_data
from scraper.pubmed import fetch_pubmed_articles
from scraper.google_patents import extract_publication_abstracts
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
            "pubmed_scraped_data": <PubMed articles>,
            "google_patent_abstracts": <Google patent abstracts>
        }
    """
    print("---running scrapper tool-------")
    
    try:
        google_patent_query_result = state.get("patents_query_result")
        print("patents_query_result:",google_patent_query_result)

        publication_numbers = []
        if google_patent_query_result:
            publication_numbers = extract_publication_numbers(google_patent_query_result)
            

        print("publication numbers:",publication_numbers)
        google_patent_abstracts = []
        if len(publication_numbers) > 0:
            google_patent_abstracts = extract_publication_abstracts(publication_numbers)
        else:
            print("No publication numbers found, skipping abstract extraction")
        
        print(google_patent_abstracts)

        output_file = "output/google_patent_abstracts.json"
        safe_json_dump(google_patent_abstracts, output_file, "Google Patent Abstracts")

        target = state.get("target")
        if not target:
            print("Warning: Missing target in state, cannot proceed with scraping")
            return {
                **state,
                "clinical_trials_scraped_data": [],
                "euctr_scraped_data": [],
                "pubmed_scraped_data": [],
                "google_patent_abstracts": google_patent_abstracts,
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
            "pubmed_scraped_data": pubmed_scraped_data,
            "google_patent_abstracts": google_patent_abstracts
        }
    
    except Exception as e:
        print(f"Critical error in scraper tool: {str(e)}")
        return {
            **state,
            "clinical_trials_scraped_data": [],
            "euctr_scraped_data": [],
            "pubmed_scraped_data": [],
            "google_patent_abstracts": [],
            "message": f"Critical error in scraper tool: {str(e)}",
            "error": str(e)
        }


def extract_publication_numbers(data):
    """Extract publication numbers with graceful error handling."""
    try:
        if not data:
            print("No data provided for publication number extraction")
            return []
        
        # Use graceful JSON parsing with fallback
        fallback_template = None  
        
        parsed_data = graceful_json_parse(data, fallback_template, "Publication Numbers Extraction")
        
        if not isinstance(parsed_data, list):
            print("Warning: Parsed data is not a list, trying to convert")
            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
            else:
                print("Could not convert to list, returning empty list")
                return []
        
        # Extract publication numbers with error handling
        publication_numbers = []
        for item in parsed_data:
            if isinstance(item, dict):
                pub_num = item.get("publication_number")
                if pub_num and pub_num != "Unknown":
                    publication_numbers.append(pub_num)
                else:
                    print(f"Warning: Missing or invalid publication number in item: {item}")
            else:
                print(f"Warning: Invalid item type in publication data: {type(item)}")
        
        print(f"Successfully extracted {len(publication_numbers)} publication numbers")
        return publication_numbers

    except Exception as e:
        print(f"Error extracting publication numbers: {str(e)}")
        return []


def extract_publication(data):
    """Extract publication numbers with enhanced error handling (legacy function)."""
    try:
        if not data:
            print("No data provided for publication extraction")
            return []
        
        # Use graceful JSON parsing with fallback
        fallback_template = None  
        
        parsed_data = graceful_json_parse(data, fallback_template, "Publication Extraction")
        
        if not isinstance(parsed_data, list):
            print("Warning: Parsed data is not a list, trying to convert")
            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
            else:
                print("Could not convert to list, returning empty list")
                return []
        
        publication_numbers = []
        for item in parsed_data:
            if isinstance(item, dict):
                pub_num = item.get("publication_number")
                if pub_num and pub_num != "Unknown":
                    publication_numbers.append(pub_num)
                else:
                    print(f"Warning: Missing or invalid publication number in item: {item}")
            else:
                print(f"Warning: Invalid item type in publication data: {type(item)}")
        
        print(f"Successfully extracted {len(publication_numbers)} publication numbers")
        return publication_numbers

    except Exception as e:
        print(f"Error extracting publication numbers: {str(e)}")
        return []


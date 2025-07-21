import urllib.parse
import requests
import json
from bs4 import BeautifulSoup
from typing import List, Dict

# def generate_patent_query_url(query: str, page: int = 0) -> str:
#     peid = "6398245427190%3A1e%3A918d9128"
#     encoded_query = urllib.parse.quote(f"inventor={query}&oq={query}&page={page}")
#     return f"https://patents.google.com/xhr/query?url={encoded_query}&exp=&tags=&peid={peid}"


def generate_patent_full_text_url(publication_number: str) -> str:
    return f"https://patents.google.com/xhr/result?id=patent%2F{publication_number}%2Fen&qs=q%3D%28CD47%29%26oq%3DCD47&exp=&peid=639825a2828e8%3A74%3A7131b2f6"

def get_patent_query_result(query: str) -> str:
    url = generate_patent_query_url(query)
    response = requests.get(url)
    return response.text

def get_patent_abstract(publication_number: str):
    print("Getting abstracts for publication numbers:", publication_number)
    url = generate_patent_full_text_url(publication_number)
    print("fetching patent abstract for:",url)
    response = requests.get(url)
    return response.text

def extract_publication_abstracts(publication_numbers):
    """
    Extract Abstract, Background, Summary, and Description for a list of publication numbers,
    save them in a text file (patent_abstracts.txt) and a JSON file (patent_data.json).
    """
    try:
        patent_data = []
        for publication_number in publication_numbers:
            print(f"Extracting sections for publication number: {publication_number}")
            
            # Fetch HTML content
            html_response = get_patent_abstract(publication_number)
            if not html_response:
                print(f"Warning: No HTML content retrieved for {publication_number}")
                patent_data.append({
                    "Publication Number": publication_number,
                    "URL": f"https://patents.google.com/patent/{publication_number}",
                    "Title": None,
                    "Abstract": {
                        "abstract": None,
                        "background": None,
                        "summary": None,
                        "description": None
                    }
                })
                continue

            # Extract sections
            # extract_abstract returns (sections_dict, title)
            result = extract_abstract_from_html(html_response)
            if result:
                sections, title = result
                patent_data.append({
                    "Publication Number": publication_number,
                    "URL": f"https://patents.google.com/patent/{publication_number}",
                    "Title": title,
                    "Abstract": sections
                })
            else:
                print(f"Warning: No sections extracted for {publication_number}")
                patent_data.append({
                    "Publication Number": publication_number,
                    "URL": f"https://patents.google.com/patent/{publication_number}",
                    "Abstract": {
                        "abstract": None,
                        "background": None,
                        "summary": None,
                        "description": None
                    }
                })
                
        # Save to JSON file (patent_data.json)
        with open("output/scraper-results/patent_abstracts.json", "w", encoding="utf-8") as file:
            json.dump(patent_data, file, indent=4, ensure_ascii=False)

        print("✅ Patent data saved to patent_abstracts.txt and patent_data.json")
        return patent_data

    except Exception as e:
        print(f"Error extracting patent data: {e}")
        return None

def extract_abstract_from_html(html_content):
    """
    Extract Abstract, Background, Summary, and Description from an HTML response using BeautifulSoup.
    Returns a dictionary with the extracted sections.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        title = None
        result = {
            'abstract': None,
            'background': None,
            'summary': None,
            'description': None
        }

        # Title
        span_title = soup.select_one('article.result span[itemprop="title"]')
        if span_title:
            title = span_title.get_text(strip=True)
        else:
            print("Warning: Title not found")

        # Abstract
        abstract_section = soup.select_one('section[itemprop="abstract"] div.abstract')
        if abstract_section:
            result['abstract'] = abstract_section.get_text(strip=True)
        else:
            print("Warning: Abstract section not found")

        # Background
        background_paras = soup.select('section[itemprop="description"] div.description-paragraph[id="p-0004"], '
                                      'section[itemprop="description"] div.description-paragraph[id="p-0005"]')
        if background_paras:
            result['background'] = ' '.join([para.get_text(strip=True) for para in background_paras])
        else:
            # Fallback: Look for <heading id="h-0003">BACKGROUND</heading> and following paragraphs
            background_heading = soup.find('heading', id='h-0003')
            if background_heading:
                background_text = []
                next_elem = background_heading.find_next()
                while next_elem and next_elem.name == 'div' and 'description-paragraph' in next_elem.get('class', []):
                    background_text.append(next_elem.get_text(strip=True))
                    next_elem = next_elem.find_next()
                result['background'] = ' '.join(background_text) if background_text else None
            else:
                print("Warning: Background section not found")

        # Summary
        summary_para = soup.select_one('section[itemprop="description"] div.description-paragraph[id="p-0006"]')
        if summary_para:
            result['summary'] = summary_para.get_text(strip=True)
        else:
            # Fallback: Look for <heading id="h-0004">SUMMARY</heading> and following paragraph
            summary_heading = soup.find('heading', id='h-0004')
            if summary_heading:
                summary_para = summary_heading.find_next('div', class_='description-paragraph')
                result['summary'] = summary_para.get_text(strip=True) if summary_para else None
            else:
                print("Warning: Summary section not found")

        # Description
        # description_section = soup.select_one('section[itemprop="description"] div[itemprop="content"]')
        # if description_section:
        #     result['description'] = description_section.get_text(strip=True)
        # else:
        #     # Fallback: Extract all text under <section itemprop="description">
        #     description_section = soup.select_one('section[itemprop="description"]')
        #     result['description'] = description_section.get_text(strip=True) if description_section else None
        #     if not result['description']:
        #         print("Warning: Description section not found")

        return result, title

    except Exception as e:
        print(f"Error processing HTML: {e}")
        return None

def generate_patent_query_url(query: str, page: int = 0) -> str:
    peid = "6398245427190%3A1e%3A918d9128"
    encoded_query = urllib.parse.quote(f"inventor={query}&oq={query}&page={page}")
    return f"https://patents.google.com/xhr/query?url={encoded_query}&exp=&tags=&peid={peid}"

def get_patent_query_results(query: str) -> list:
    try:
        # First request to get total number of pages
        first_url = generate_patent_query_url(query, page=0)
        first_response = requests.get(first_url)
        first_response.raise_for_status()

        first_data = first_response.json()
        total_pages = first_data.get("results", {}).get("total_num_pages", 1)
        all_results = [first_data]

        # Fetch remaining pages
        for page in range(1, total_pages):
            try:
                url = generate_patent_query_url(query, page=page)
                response = requests.get(url)
                response.raise_for_status()
                all_results.append(response.json())
            except requests.exceptions.RequestException as e:
                print(f"[WARNING] Failed to fetch page {page}: {e}")
            except ValueError:
                print(f"[WARNING] JSON decode error on page {page}")

        return clean_patent_results(all_results)

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch first page: {e}")
    except ValueError:
        print("[ERROR] JSON decode error on first page")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

    return []  # Return empty list on failure

def clean_patent_results(patents: List[Dict]) -> List[Dict]:
    """
    Clean and extract relevant fields from raw patent entries.

    Args:
        patents (List[Dict]): Raw patent data from Google Patents API.

    Returns:
        List[Dict]: Cleaned patent data with title, publication_number, status, assignee, etc.
    """
    cleaned = []

    for patent_data in patents:
        clusters = patent_data.get("results", {}).get("cluster", [])
        for cluster in clusters:
            results = cluster.get("result", [])
            for result in results:
                patent = result.get("patent", {})
                cleaned.append({
                    "title": patent.get("title", "").strip(),
                    "publication_number": patent.get("publication_number", ""),
                    "assignee": patent.get("assignee", ""),
                    "status": "Granted" if patent.get("grant_date") else "Pending",
                })

    return cleaned

# def main():
#     query = "CD47"
#     patents_query_result = get_patent_query_results(query)
#     with open("patents_query_result.json", "w") as f:
#         json.dump(patents_query_result, f)

# if __name__ == "__main__":
#     main()
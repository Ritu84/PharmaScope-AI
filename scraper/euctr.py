import requests
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_euctr_data(target: str):
    url = f"https://www.clinicaltrialsregister.eu/ctr-search/rest/download/summary?query={target}&mode=current_page"
    try:
        response = requests.get(url, headers={"User-Agent": "MyApp/1.0 (adarsh@example.com)"})
        response.raise_for_status()
        output_dir = "output/scraper-results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "euctr.json")
        with open(output_path, "w") as f:
            f.write(response.text)
        return response.json
    except requests.exceptions.RequestException as e:
        logging.error(f"EUCTR API request failed: {e}")
        return []

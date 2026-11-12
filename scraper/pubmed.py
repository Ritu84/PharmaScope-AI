from pymed_paperscraper import PubMed
import json
import os

def fetch_pubmed_articles(query, max_results=500):
    pubmed = PubMed(tool="PharmaScope-AI", email="rsharma114962@gmail.com")
    results = pubmed.query(query, max_results=max_results)
    articles = []
    for article in results:
        # Use toJSON() to get a JSON-serializable string representation
        if hasattr(article, "toJSON"):
            articles.append(json.loads(article.toJSON()))
        else:
            articles.append(str(article))
        
    abstracts = [{"abstract": item["abstract"]} for item in articles if "abstract" in item]
    output_dir = "output/scraper-results"
    os.makedirs(output_dir, exist_ok=True)
    # Keep PMID, title and date too, so the RAG index can cite each abstract.
    records = [
        {k: item.get(k) for k in ("pubmed_id", "title", "publication_date", "abstract")}
        for item in articles if isinstance(item, dict) and item.get("abstract")
    ]
    with open(os.path.join(output_dir, "pubmed_records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(output_dir, "pubmed.json"), "w", encoding="utf-8") as f:
        json.dump(abstracts, f, indent=2, ensure_ascii=False)

    print("Abstracts saved to abstracts.json")

    return abstracts

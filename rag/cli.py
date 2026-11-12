"""Command line for the RAG layer.

    python -m rag.cli index CD47            # scrape PubMed + ClinicalTrials.gov, then build the vector index
    python -m rag.cli ask CD47 "Which CD47 antibodies reached phase 3?"
"""
import json
import os
import sys

from rag.graph import ask, build_index

RESULTS = "output/scraper-results"


def scrape_and_index(target: str) -> int:
    from scraper.clinicaltrials import fetch_clinicaltrials_data
    from scraper.pubmed import fetch_pubmed_articles

    fetch_pubmed_articles(target)
    trials = fetch_clinicaltrials_data(target) or {}
    with open(os.path.join(RESULTS, "pubmed_records.json")) as f:
        articles = json.load(f)
    return build_index(target, articles, trials)


def main(argv):
    if len(argv) >= 2 and argv[0] == "index":
        print(f"Indexed {scrape_and_index(argv[1])} chunks for {argv[1]}")
    elif len(argv) >= 3 and argv[0] == "ask":
        r = ask(argv[1], " ".join(argv[2:]))
        print(r["answer"])
        print("\nSources:", ", ".join(r["citations"]) or "none")
        if r["unsupported_citations"]:
            print("WARNING, cited but not retrieved:", ", ".join(r["unsupported_citations"]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])

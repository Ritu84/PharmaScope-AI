"""Turn scraped PubMed articles and ClinicalTrials.gov studies into documents for the RAG index.

Each document keeps a stable source id (PMID:<id> or NCT id) so answers can cite it.
"""
import re
from typing import Any, Dict, List

WORDS_PER_CHUNK = 180
OVERLAP_WORDS = 40


def pubmed_documents(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for a in articles or []:
        abstract = (a.get("abstract") or "").strip()
        pmid = str(a.get("pubmed_id") or a.get("pmid") or "").split("\n")[0].strip()
        if not abstract or not pmid:
            continue
        docs.append({
            "source_id": f"PMID:{pmid}",
            "source": "pubmed",
            "title": (a.get("title") or "").strip(),
            "year": str(a.get("publication_date") or "")[:4],
            "phase": "",
            "text": abstract,
        })
    return docs


def strip_tags(text: str) -> str:
    """ClinicalTrials.gov highlights search terms with <mark> tags."""
    return re.sub(r"<[^>]+>", "", text or "")


def _studies(data: Any) -> List[Dict[str, Any]]:
    """The ClinicalTrials.gov internal API returns {"hits": [{"study": {...}}]}; the public v2 API returns {"studies": [...]}."""
    if not isinstance(data, dict):
        return []
    if "hits" in data:
        return [h.get("study", h) for h in data["hits"]]
    return data.get("studies", [])


def trial_documents(data: Any) -> List[Dict[str, Any]]:
    docs = []
    for s in _studies(data):
        p = s.get("protocolSection", s)
        ident = p.get("identificationModule", {})
        nct = ident.get("nctId")
        if not nct:
            continue
        status = p.get("statusModule", {})
        design = p.get("designModule", {})
        sponsor = p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "")
        arms = p.get("armsInterventionsModule", {}).get("interventions", [])
        outcomes = p.get("outcomesModule", {}).get("primaryOutcomes", [])
        conditions = p.get("conditionsModule", {}).get("conditions", [])
        phase = ", ".join(design.get("phases", []))
        title = strip_tags(ident.get("briefTitle", ""))
        text = ". ".join(x for x in [
            title,
            f"Phase: {phase}" if phase else "",
            f"Status: {status.get('overallStatus', '')}",
            f"Sponsor: {sponsor}" if sponsor else "",
            f"Conditions: {', '.join(conditions)}" if conditions else "",
            "Interventions: " + "; ".join(f"{i.get('type', '')} {i.get('name', '')}".strip() for i in arms) if arms else "",
            "Primary outcomes: " + "; ".join(o.get("measure", "") for o in outcomes) if outcomes else "",
            f"Start: {status.get('startDateStruct', {}).get('date', '')}",
        ] if x)
        docs.append({
            "source_id": nct,
            "source": "clinicaltrials",
            "title": title,
            "year": status.get("startDateStruct", {}).get("date", "")[:4],
            "phase": phase,
            "text": strip_tags(text),
        })
    return docs


def chunk_text(text: str, size: int = WORDS_PER_CHUNK, overlap: int = OVERLAP_WORDS) -> List[str]:
    """Split on sentence ends into chunks of about `size` words, repeating the last `overlap` words."""
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    chunks, current = [], []
    for sentence in sentences:
        words = sentence.split()
        if current and len(current) + len(words) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for d in docs:
        for i, piece in enumerate(chunk_text(d["text"])):
            chunks.append({**d, "chunk_id": f"{d['source_id']}#{i}", "text": piece})
    return chunks

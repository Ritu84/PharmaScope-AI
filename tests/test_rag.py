"""Offline tests for the RAG layer: no network, a fake embedder and a fake LLM."""
import hashlib
import math

from chromadb import EmbeddingFunction

from rag.answer import answer, cited_ids
from rag.documents import chunk_documents, chunk_text, pubmed_documents, strip_tags, trial_documents
from rag.store import index_chunks, retrieve


class BagOfWords(EmbeddingFunction):
    """Deterministic stand-in for Gemini: hashed word counts, so similar text gets similar vectors."""

    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * 64
            for w in text.lower().split():
                v[int(hashlib.md5(w.strip(".,").encode()).hexdigest(), 16) % 64] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out

    @staticmethod
    def name():
        return "bag-of-words-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return BagOfWords()


TRIALS = {"hits": [{"study": {"protocolSection": {
    "identificationModule": {"nctId": "NCT01234567", "briefTitle": "Anti-<mark>CD47</mark> magrolimab in AML"},
    "statusModule": {"overallStatus": "TERMINATED", "startDateStruct": {"date": "2021-03"}},
    "designModule": {"phases": ["PHASE3"]},
    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Gilead Sciences"}},
}}}]}
ARTICLES = [
    {"pubmed_id": "111\n222", "title": "Lemzoparlimab safety", "publication_date": "2024-05-01",
     "abstract": "Lemzoparlimab binds CD47 with low red blood cell binding and showed manageable anemia."},
    {"pubmed_id": "333", "title": "No abstract", "abstract": ""},
]


def test_pubmed_documents_use_first_pmid_and_skip_empty_abstracts():
    docs = pubmed_documents(ARTICLES)
    assert [d["source_id"] for d in docs] == ["PMID:111"]
    assert docs[0]["year"] == "2024"


def test_trial_documents_strip_highlight_tags_and_keep_phase_and_sponsor():
    [doc] = trial_documents(TRIALS)
    assert doc["source_id"] == "NCT01234567"
    assert "<mark>" not in doc["text"] and "<mark>" not in doc["title"]
    assert "Phase: PHASE3" in doc["text"] and "Gilead Sciences" in doc["text"]
    assert strip_tags("a <mark>b</mark>") == "a b"


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} has some words." for i in range(100))
    chunks = chunk_text(text, size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 56 for c in chunks)  # a sentence is never split, so a chunk may run a few words over
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_retrieval_finds_the_relevant_source_and_filters_by_source():
    chunks = chunk_documents(pubmed_documents(ARTICLES) + trial_documents(TRIALS))
    col = index_chunks("CD47", chunks, path=None, embedder=BagOfWords())
    assert retrieve(col, "magrolimab terminated phase 3 trial Gilead", k=1)[0]["source_id"] == "NCT01234567"
    only_pubmed = retrieve(col, "magrolimab terminated phase 3 trial Gilead", k=2, source="pubmed")
    assert {c["source_id"] for c in only_pubmed} == {"PMID:111"}


def test_cited_ids_handles_grouped_and_single_citations():
    text = "A [NCT01234567, PMID:111]. B [PMID: 222]. C [see table]."
    assert cited_ids(text) == ["NCT01234567", "PMID:111", "PMID:222"]


def test_answer_flags_citations_that_were_not_retrieved():
    chunks = [{"source_id": "PMID:111", "source": "pubmed", "title": "t", "year": "2024", "phase": "", "text": "x"}]
    fake = lambda prompt: {"text": "Low RBC binding [PMID:111]; approved in 2020 [PMID:999].", "prompt_tokens": 10}
    r = answer("CD47", "q", chunks, llm=fake)
    assert r["citations"] == ["PMID:111"]
    assert r["unsupported_citations"] == ["PMID:999"]


def test_answer_without_sources_does_not_call_the_model():
    def boom(prompt):
        raise AssertionError("LLM should not be called")
    assert answer("CD47", "q", [], llm=boom)["citations"] == []

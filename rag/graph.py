"""LangGraph flows for RAG: build the index from scraped data, and retrieve -> answer for a question."""
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from rag.answer import answer
from rag.documents import chunk_documents, pubmed_documents, trial_documents
from rag.store import get_collection, index_chunks, retrieve


def build_index(target: str, pubmed_articles: List[Dict[str, Any]], trials: Any) -> int:
    docs = pubmed_documents(pubmed_articles) + trial_documents(trials)
    return index_chunks(target, chunk_documents(docs)).count()


class AskState(TypedDict, total=False):
    target: str
    question: str
    k: int
    source: Optional[str]
    chunks: List[Dict[str, Any]]
    result: Dict[str, Any]


def _retrieve(state: AskState) -> AskState:
    col = get_collection(state["target"])
    return {"chunks": retrieve(col, state["question"], state.get("k", 5), state.get("source"))}


def _answer(state: AskState) -> AskState:
    return {"result": answer(state["target"], state["question"], state["chunks"])}


def build_ask_flow():
    g = StateGraph(AskState)
    g.add_node("retrieve", _retrieve)
    g.add_node("answer", _answer)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()


def ask(target: str, question: str, k: int = 5, source: Optional[str] = None) -> Dict[str, Any]:
    state = build_ask_flow().invoke({"target": target, "question": question, "k": k, "source": source})
    return {**state["result"], "sources": state["chunks"]}

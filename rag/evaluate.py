"""Evaluate the RAG layer on a target that is already indexed.

    python -m rag.evaluate CD47 --n 30

1. Retrieval: for N randomly sampled chunks, Gemini writes a question that chunk answers (paraphrased, not copied).
   We then check whether retrieval brings back that chunk's source in the top 1 / top 5 (hit@1, hit@5, MRR).
2. Answers: for the first 10 questions we generate an answer and count answers with at least one valid citation
   and citations that point at sources that were never retrieved (hallucinated citations).
3. Cost: tokens sent per question with RAG vs. sending every abstract and trial record for the target.
"""
import argparse
import json
import os
import random
import time

from rag.answer import answer, format_sources, gemini_chat
from rag.store import get_collection, retrieve
from utils.tokenize import count_tokens

QUESTION_PROMPT = """Here is a passage from a biomedical source about the drug target {target}.
Write ONE specific question that this passage answers, as a researcher would ask it.
Paraphrase: do not copy phrases longer than 3 words from the passage, and do not mention "the passage" or "the study".
Reply with the question only.

Passage:
{text}"""

PAUSE = 4  # seconds between chat calls, to stay inside the free-tier rate limit


def run(target: str, n: int, k: int, seed: int) -> dict:
    col = get_collection(target)
    data = col.get(include=["documents", "metadatas"])
    items = list(zip(data["ids"], data["documents"], data["metadatas"]))
    random.Random(seed).shuffle(items)
    sample = items[:n]

    rows = []
    for cid, text, meta in sample:
        q = gemini_chat(QUESTION_PROMPT.format(target=target, text=text))["text"].strip()
        time.sleep(PAUSE)
        hits = retrieve(col, q, k=k)
        ranked = list(dict.fromkeys(h["source_id"] for h in hits))
        rank = ranked.index(meta["source_id"]) + 1 if meta["source_id"] in ranked else None
        rows.append({"question": q, "expected": meta["source_id"], "retrieved": ranked, "rank": rank, "hits": hits})

    answers = []
    for r in rows[:10]:
        a = answer(target, r["question"], r["hits"])
        time.sleep(PAUSE)
        answers.append({"question": r["question"], "answer": a["answer"], "citations": a["citations"],
                        "unsupported_citations": a["unsupported_citations"], "prompt_tokens": a["prompt_tokens"]})

    full_context_tokens = count_tokens(format_sources([{**m, "text": d} for _, d, m in items]))
    rag_tokens = [a["prompt_tokens"] for a in answers if a["prompt_tokens"]]
    avg_rag = sum(rag_tokens) / len(rag_tokens) if rag_tokens else 0
    summary = {
        "target": target,
        "chunks_indexed": len(items),
        "questions": len(rows),
        f"hit@1": round(sum(1 for r in rows if r["rank"] == 1) / len(rows), 3),
        f"hit@{k}": round(sum(1 for r in rows if r["rank"]) / len(rows), 3),
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows), 3),
        "answers_checked": len(answers),
        "answers_with_valid_citation": sum(1 for a in answers if a["citations"]),
        "answers_with_unsupported_citation": sum(1 for a in answers if a["unsupported_citations"]),
        "avg_prompt_tokens_with_rag": round(avg_rag),
        "tokens_if_all_sources_sent": full_context_tokens,
        "token_reduction": round(1 - avg_rag / full_context_tokens, 3) if avg_rag else None,
    }
    os.makedirs("output/rag_eval", exist_ok=True)
    with open(f"output/rag_eval/{target}.json", "w") as f:
        json.dump({"summary": summary, "retrieval": [{k2: v for k2, v in r.items() if k2 != "hits"} for r in rows],
                   "answers": answers}, f, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    print(json.dumps(run(args.target, args.n, args.k, args.seed), indent=2))

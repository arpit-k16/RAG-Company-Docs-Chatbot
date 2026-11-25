"""Small evaluation harness: retrieval hit-rate + answer correctness on a golden set.

    python evaluate.py            # full eval (retrieval + LLM answers)
    python evaluate.py --retrieval-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import settings
from rag import RAGChatbot

GOLDEN = Path(__file__).parent / "data" / "eval" / "golden_set.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true")
    args = p.parse_args()

    bot = RAGChatbot.from_settings(settings)
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    hits = correct = 0
    t0 = time.time()

    for c in cases:
        q, expected = c["question"], c.get("expected_source")
        retrieved = [d.metadata.get("source") for d, _ in bot.retrieve(q)]
        hit = expected is None or expected in retrieved
        hits += hit
        line = f"{'✅' if hit else '❌'} retrieval  {q[:60]:<60} → {retrieved[:3]}"

        if not args.retrieval_only:
            ans = bot.ask(q)
            text = ans.text.lower()
            if expected is None:  # out-of-scope question: the bot should say it doesn't know
                ok = any(k in text for k in ("couldn't find", "could not find", "don't know", "not mentioned"))
            else:
                ok = all(k.lower() in text for k in c["must_include"])
            correct += ok
            line += f"\n   {'✅' if ok else '❌'} answer: {ans.text[:140]!r}"
        print(line)

    n = len(cases)
    print(f"\nRetrieval hit rate: {hits}/{n} = {hits / n:.0%}")
    if not args.retrieval_only:
        print(f"Answer accuracy:    {correct}/{n} = {correct / n:.0%}")
    print(f"Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""Chat with your company docs in the terminal.

    python cli.py
    python cli.py -q "What is our leave policy?"
"""
from __future__ import annotations

import argparse

from config import settings
from rag import RAGChatbot
from rag.chatbot import to_messages


def print_sources(sources) -> None:
    if sources:
        print("\n  Sources:")
        for s in sources:
            print(f"   [{s.number}] {s.label}  (relevance {s.score:.2f})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", help="Ask one question and exit")
    args = p.parse_args()

    bot = RAGChatbot.from_settings(settings)

    if args.question:
        ans = bot.ask(args.question)
        print(ans.text)
        print_sources(ans.sources)
        return

    print(f"🤖 {settings.company_name} AI Assistant — ask about company docs. Type 'exit' to quit.\n")
    turns: list[dict] = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ":q"}:
            break
        if not q:
            continue
        sources, tokens = bot.stream(q, to_messages(turns))
        print("Bot: ", end="", flush=True)
        answer = "".join(print(t, end="", flush=True) or t for t in tokens)
        print()
        print_sources(sources)
        print()
        turns += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()

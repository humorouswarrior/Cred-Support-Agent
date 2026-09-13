"""Interactive terminal chat with the Cred support agent - for manual testing.

    python scripts/chat.py                              # in-process, no server needed
    python scripts/chat.py --ws ws://127.0.0.1:8000/ws/chat   # through the running API

Commands inside the chat:
    /debug          toggle details (tools, citations, guardrails, review, budget)
    /session NAME   switch to (or create) a named conversation
    /reset          forget the current conversation
    /help           show this list
    /quit           exit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
import uuid

import _common  # puts the project root on sys.path

assert _common.PROJECT_ROOT.exists()

HELP = __doc__.split("Commands inside the chat:")[1].rstrip()

def format_answer(answer: str, label_width: int = 7) -> str:
    """Wrap the reply to the window, keeping its paragraphs, so it stays readable."""
    indent = " " * label_width
    width = max(shutil.get_terminal_size((100, 24)).columns - 2, 50)
    blocks = []
    for number, paragraph in enumerate(answer.split("\n\n")):
        blocks.append(
            textwrap.fill(
                " ".join(paragraph.split()),
                width=width,
                initial_indent="" if number == 0 else indent,
                subsequent_indent=indent,
            )
        )
    return "\n\n".join(blocks)


def _colours_enabled() -> bool:
    """Colour only on a real terminal, and never when NO_COLOR is set.

    On Windows, an empty os.system call switches the console into the mode that
    understands ANSI colour codes (Windows 10 and later).
    """
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        os.system("")
    return True


_COLOUR = _colours_enabled()
DIM = "\033[2m" if _COLOUR else ""
BOLD = "\033[1m" if _COLOUR else ""
CYAN = "\033[36m" if _COLOUR else ""
YELLOW = "\033[33m" if _COLOUR else ""
RED = "\033[31m" if _COLOUR else ""
RESET = "\033[0m" if _COLOUR else ""


def _details(fields: dict) -> None:
    for key, value in fields.items():
        if value in (None, [], {}, ""):
            continue
        print(f"{DIM}  {key:<22} {value}{RESET}")


def run_in_process(debug: bool) -> None:
    from cred_support_agent.safety.governance import BudgetExceeded
    from cred_support_agent.service import ask, conversation_turns, reset_conversation

    session = f"chat-{uuid.uuid4().hex[:6]}"
    print(f"{DIM}loading the agent (first answer takes a few seconds)...{RESET}")

    while True:
        try:
            text = input(f"\n{BOLD}you [{session}]>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return
        if text == "/help":
            print(HELP)
            continue
        if text == "/debug":
            debug = not debug
            print(f"{DIM}debug {'on' if debug else 'off'}{RESET}")
            continue
        if text == "/reset":
            reset_conversation(session)
            print(f"{DIM}conversation '{session}' cleared{RESET}")
            continue
        if text.startswith("/session"):
            parts = text.split(maxsplit=1)
            session = parts[1] if len(parts) > 1 else f"chat-{uuid.uuid4().hex[:6]}"
            print(f"{DIM}now in session '{session}' ({conversation_turns(session) // 2} earlier turns){RESET}")
            continue

        try:
            result = ask(text, session_id=session, endpoint="chat-cli")
        except BudgetExceeded as exc:
            print(f"{RED}rejected by the runtime budget cap:{RESET} {exc}")
            continue

        response = result["response"]
        colour = YELLOW if response.answer_type == "refusal" else CYAN
        print(f"{colour}agent>{RESET} {format_answer(response.answer)}")
        if debug:
            review = result.get("review") or {}
            run = result["result"]
            _details(
                {
                    "answer_type": response.answer_type,
                    "resolved_question": result.get("resolved_question")
                    if result.get("resolved_question") != text
                    else None,
                    "tools_used": response.tools_used,
                    "citations": response.citations,
                    "record_id": response.record_id,
                    "escalation": response.escalation_recommended if response.record_id else None,
                    "retrieval_confidence": response.confidence or None,
                    "guardrail_flags": response.guardrail_flags,
                    "review": (
                        f"approved={review.get('approved')} - {review.get('reason')}" if review else None
                    ),
                    "tokens_used": run.budget.get("total_tokens"),
                    "trace_id": result["trace_id"],
                }
            )


def run_over_websocket(url: str, debug: bool) -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    try:
        socket = connect(url, open_timeout=10)
    except OSError as exc:
        print(f"{RED}could not connect to {url}: {exc}{RESET}")
        print("start the server first:  uvicorn cred_support_agent.api:app --port 8000")
        sys.exit(1)

    with socket:
        ready = json.loads(socket.recv())
        print(f"{DIM}connected, session '{ready['session_id']}'{RESET}")
        while True:
            try:
                text = input(f"\n{BOLD}you>{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                continue
            if text in ("/quit", "/exit"):
                return
            if text == "/debug":
                debug = not debug
                print(f"{DIM}debug {'on' if debug else 'off'}{RESET}")
                continue
            if text.startswith("/"):
                print(f"{DIM}over WebSocket only /debug and /quit are available{RESET}")
                continue
            try:
                socket.send(json.dumps({"query": text}))
                frame = json.loads(socket.recv(timeout=120))
            except ConnectionClosed:
                print(f"{RED}server closed the connection{RESET}")
                return

            if frame["type"] != "answer":
                print(f"{RED}{frame['type']}:{RESET} {frame.get('detail')}")
                continue
            colour = YELLOW if frame["answer_type"] == "refusal" else CYAN
            print(f"{colour}agent>{RESET} {format_answer(frame['answer'])}")
            if debug:
                shown = {k: frame.get(k) for k in (
                    "answer_type", "resolved_question", "tools_used", "citations",
                    "escalation_recommended", "guardrail_flags", "review_approved", "trace_id",
                )}
                if shown["resolved_question"] == text:
                    shown["resolved_question"] = None
                if not frame.get("tools_used") or "check_loan_application_status" not in frame["tools_used"]:
                    shown["escalation_recommended"] = None
                _details(shown)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the Cred support agent.")
    parser.add_argument("--ws", metavar="URL", help="talk to a running server over WebSocket")
    parser.add_argument("--debug", action="store_true", help="start with details shown")
    args = parser.parse_args()

    print(f"{BOLD}Cred support agent{RESET} {DIM}- MOCK_LLM, no API keys. /help for commands.{RESET}")
    if args.ws:
        run_over_websocket(args.ws, args.debug)
    else:
        run_in_process(args.debug)


if __name__ == "__main__":
    main()

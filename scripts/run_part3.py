"""Part 3 - FastAPI deployment, structured logging, evaluation (Tasks 11-13)."""

from __future__ import annotations

import json
import re

from _common import banner, transcript

from cred_support_agent.evaluation import judge
from cred_support_agent.observability import clear_log, read_log

RAW_PII = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b|\b\d{9,18}\b|\b\d{4}[ -]\d{4}[ -]\d{4}\b")


def main() -> None:
    from fastapi.testclient import TestClient

    from cred_support_agent.api import app

    client = TestClient(app)
    clear_log()

    with transcript("task11_fastapi.txt"):
        banner("TASK 11 - FASTAPI: HTTP ENDPOINTS + WEBSOCKET, PYDANTIC MODELS")
        print("registered routes (response model):")
        for route in app.routes:
            methods = getattr(route, "methods", None)
            label = ",".join(sorted(methods)) if methods else "WEBSOCKET"
            model = getattr(route, "response_model", None)
            print(f"  {label:<16} {route.path:<28} {getattr(model, '__name__', '')}")

        print("\n--- GET /health ---")
        print(json.dumps(client.get("/health").json(), indent=2))

        print("\n--- POST /ask (AskRequest -> AskResponse) ---")
        response = client.post("/ask", json={"query": "What are the KYC document requirements?", "session_id": "http-1"})
        print(f"  HTTP {response.status_code}   X-Trace-Id: {response.headers['x-trace-id']}")
        print(json.dumps(response.json(), indent=2)[:1800])

        print("\n--- POST /ask (status lookup) ---")
        body = client.post("/ask", json={"query": "What is the status of CRED-LN-0009?", "session_id": "http-1"}).json()
        print(f"  answer_type={body['response']['answer_type']}  review approved={(body.get('review') or {}).get('approved')}")
        print(f"  answer: {body['response']['answer']}")

        print("\n--- POST /add-document (AddDocumentRequest -> AddDocumentResponse) ---")
        before = client.post("/ask", json={"query": "Can I take a top-up on my existing loan?", "session_id": "http-2"}).json()
        print(f"  before: {before['response']['answer'][:90]}")
        response = client.post(
            "/add-document",
            json={
                "doc_id": "KB-101",
                "topic": "loan_top_up_policy",
                "title": "Top-up loan policy",
                "text": (
                    "A Cred borrower may apply for a top-up on an existing loan once twelve "
                    "consecutive instalments have been paid on time. The top-up is capped at "
                    "40 percent of the original sanctioned amount and is priced at the same "
                    "rate as the parent loan. A top-up request is declined outright while any "
                    "instalment on the parent loan is overdue."
                ),
            },
        )
        print(f"  HTTP {response.status_code}")
        print("  " + json.dumps(response.json(), indent=2).replace("\n", "\n  "))
        after = client.post("/ask", json={"query": "Can I take a top-up on my existing loan?", "session_id": "http-3"}).json()
        print(f"  after : {after['response']['answer']}")
        from cred_support_agent.service import remove_document

        cleanup = remove_document("KB-101")
        print(f"  (demo cleanup: KB-101 removed again, chunks {cleanup['removed_chunks']}, "
              "so later tasks measure the base knowledge base)")

        print("\n--- request validation by the Pydantic models ---")
        bad = client.post("/ask", json={"query": ""})
        print(f"  empty query -> HTTP {bad.status_code}: {bad.json()['detail'][0]['msg']}")

        print("\n--- WEBSOCKET /ws/chat: multi-turn, malformed frame, abrupt disconnect ---")
        with client.websocket_connect("/ws/chat?session_id=ws-demo") as socket:
            print(f"  <- {socket.receive_json()['type']}")
            socket.send_json({"query": "What is the status of CRED-LN-0009?"})
            frame = socket.receive_json()
            print(f"  -> turn 1 [{frame['answer_type']}] {frame['answer'][:150]}")
            socket.send_text("this is not valid json")
            frame = socket.receive_json()
            print(f"  -> malformed frame answered with type={frame['type']}; socket still open")
            socket.send_json({"query": "Is that one escalated?"})
            frame = socket.receive_json()
            print(f"  -> turn 2 resolved as {frame['resolved_question']!r}")
            print(f"     {frame['answer'][:150]}")
            print("  (client disconnects mid-conversation: WebSocketDisconnect is caught)")

        print("\n  after the disconnect the server keeps serving:")
        print(f"    GET /health -> HTTP {client.get('/health').status_code}")
        with client.websocket_connect("/ws/chat?session_id=ws-second") as socket:
            socket.receive_json()
            socket.send_json({"query": "What is the minimum balance requirement?"})
            print(f"    a second WebSocket client is answered: {socket.receive_json()['answer'][:100]}")

        print("\n--- oversized request rejected by the runtime budget (HTTP 413) ---")
        oversized = client.post("/ask", json={"query": "Please explain the entire Cred lending policy. " * 90})
        print(f"  HTTP {oversized.status_code}  detail: {json.dumps(oversized.json()['detail'])}")

    with transcript("task12_structured_logging.txt"):
        banner("TASK 12 - STRUCTURED JSON-LINES LOGGING (one entry per request)")
        pii_query = (
            "My PAN is ABCDE1234F, Aadhaar 4321 8765 2109 and account 001234567890123. "
            "What is the status of CRED-LN-0009?"
        )
        print(f"--- a request carrying fixed-format PII ---\n  sent: {pii_query}")
        client.post("/ask", json={"query": pii_query, "session_id": "http-pii"})

        entries = read_log()
        print(f"\n  {len(entries)} JSON-lines entries in artifacts/requests.jsonl, one per request above")
        print("\n  most recent entry, verbatim:")
        print("  " + json.dumps(entries[-1]))
        print("\n  every entry carries a trace id and timing metadata:")
        for entry in entries:
            print(
                f"    {entry['trace_id']}  {entry['endpoint']:<22} http={str(entry.get('http_status', '-')):<4} "
                f"{entry['latency_ms']:>9.2f}ms  status={entry.get('status'):<9} "
                f"stages={[s['stage'] for s in entry.get('stages', [])]}"
            )
        leaks = [e["trace_id"] for e in entries if RAW_PII.search(json.dumps(e))]
        print(f"\n  entries anywhere in the file containing raw fixed-format PII: {len(leaks)}")
        pii_entry = next(e for e in reversed(entries) if e.get("pii_masked"))
        print(f"  logged request_text : {pii_entry['request_text']}")
        print(f"  masking recorded    : {pii_entry['pii_masked']}")

    with transcript("task13_evaluation.txt"):
        banner("TASK 13 - LLM-AS-JUDGE EVALUATION, 15 QUERIES, UNDER MOCK_LLM")
        print("judge prompt (system):")
        print("  " + judge.JUDGE_SYSTEM_PROMPT.replace("\n", "\n  "))
        print(judge.report())
        print("raw scores: transcripts/task13_evaluation_scores.json")


if __name__ == "__main__":
    main()

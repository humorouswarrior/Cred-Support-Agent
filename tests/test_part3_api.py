"""Part 3 acceptance tests - FastAPI, logging, evaluation."""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from cred_support_agent.api import app
from cred_support_agent.evaluation.benchmarks import JUDGE_EVAL_QUERIES
from cred_support_agent.evaluation.judge import DIMENSIONS, evaluate_case, judge
from cred_support_agent.data.knowledge_base import REQUIRED_TOPICS
from cred_support_agent.observability import clear_log, read_log

RAW_PII = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b|\b\d{9,18}\b|\b\d{4}[ -]\d{4}[ -]\d{4}\b")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# -- Task 11 --------------------------------------------------------------


def test_at_least_two_http_endpoints_and_one_websocket():
    http_paths = {r.path for r in app.routes if getattr(r, "methods", None)}
    ws_paths = {r.path for r in app.routes if not getattr(r, "methods", None)}
    assert {"/ask", "/add-document"} <= http_paths
    assert "/ws/chat" in ws_paths


def test_ask_returns_a_schema_valid_response(client):
    response = client.post("/ask", json={"query": "What are the KYC requirements?"})
    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"]
    assert body["response"]["answer"]
    assert body["response"]["answer_type"] in {"policy", "status", "policy_and_status", "refusal"}


def test_ask_rejects_an_empty_query(client):
    assert client.post("/ask", json={"query": ""}).status_code == 422


def test_add_document_indexes_into_both_collections(client):
    response = client.post(
        "/add-document",
        json={
            "doc_id": "KB-TEST-1",
            "topic": "test_topic",
            "title": "Test policy",
            "text": (
                "A Cred sandbox policy document used only by the automated test suite. "
                "It exists so the add-document endpoint can be exercised end to end. "
                "It states that sandbox accounts carry no fees whatsoever."
            ),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["collections"]) == 2
    assert all(count > 0 for count in body["indexed_chunks"].values())


def test_websocket_multi_turn_and_disconnect_resilience(client):
    with client.websocket_connect("/ws/chat?session_id=test-ws") as socket:
        assert socket.receive_json()["type"] == "ready"
        socket.send_json({"query": "What is the status of CRED-LN-0007?"})
        first = socket.receive_json()
        assert first["type"] == "answer"
        assert first["answer"]

        socket.send_text("{not valid json")
        assert socket.receive_json()["type"] == "error"  # socket survives

        socket.send_json({"query": "Is that one escalated?"})
        second = socket.receive_json()
        assert "CRED-LN-0007" in second["resolved_question"]

    # The abrupt disconnect above must not have taken the server down.
    assert client.get("/health").status_code == 200
    with client.websocket_connect("/ws/chat?session_id=test-ws-2") as socket:
        assert socket.receive_json()["type"] == "ready"
    assert client.post("/ask", json={"query": "What are the KYC rules?"}).status_code == 200


def test_oversized_request_is_rejected_with_413(client):
    response = client.post("/ask", json={"query": "Explain the whole policy in detail. " * 100})
    assert response.status_code == 413
    assert response.json()["detail"]["reason"] == "request_token_cap"


# -- Task 12 --------------------------------------------------------------


def test_every_request_writes_one_json_line_with_trace_id_and_timing(client):
    clear_log()
    before = len(read_log())
    client.post("/ask", json={"query": "What are the interest rate slabs?"})
    entries = read_log()
    assert len(entries) == before + 1
    entry = entries[-1]
    assert entry["trace_id"].startswith("trc_")
    assert entry["latency_ms"] > 0
    assert entry["endpoint"] == "POST /ask"
    assert entry["stages"]


def test_logged_request_text_is_masked(client):
    clear_log()
    client.post(
        "/ask",
        json={
            "query": (
                "My PAN is ABCDE1234F, Aadhaar 4321 8765 2109, account 001234567890123. "
                "What is the status of CRED-LN-0007?"
            )
        },
    )
    entries = read_log()
    entry = entries[-1]
    assert "[PAN_REDACTED]" in entry["request_text"]
    assert entry["pii_masked"] == {"pan": 1, "aadhaar": 1, "bank_account": 1}


def test_api_response_reports_that_masking_fired(client):
    """Masking happens before the crew, so the flag must be carried through."""
    response = client.post(
        "/ask",
        json={"query": "My PAN is ABCDE1234F and Aadhaar 4321 8765 2109. Status of CRED-LN-0002?"},
    )
    flags = response.json()["response"]["guardrail_flags"]
    assert "pii_masked:pan" in flags
    assert "pii_masked:aadhaar" in flags
    assert read_log()[-1]["guardrail_flags"] == flags


def test_no_log_entry_anywhere_contains_raw_fixed_format_pii():
    for entry in read_log():
        assert not RAW_PII.search(json.dumps(entry)), entry["trace_id"]


def test_rejected_request_is_still_logged(client):
    clear_log()
    client.post("/ask", json={"query": "Explain the whole policy in detail. " * 100})
    entry = read_log()[-1]
    assert entry["status"] in {"error", "rejected"}
    assert entry["trace_id"]


# -- Task 13 --------------------------------------------------------------


def test_evaluation_set_has_15_queries():
    assert len(JUDGE_EVAL_QUERIES) == 15


def test_evaluation_set_covers_every_required_topic():
    topics = {case["topic"] for case in JUDGE_EVAL_QUERIES}
    missing = [t for t in REQUIRED_TOPICS if t not in topics]
    assert not missing, missing


def test_evaluation_set_has_at_least_two_out_of_scope_or_edge_cases():
    edge = [c for c in JUDGE_EVAL_QUERIES if c["expectation"] in {"refusal", "blocked"}]
    assert len(edge) >= 2


def test_judge_scores_all_four_dimensions():
    case = JUDGE_EVAL_QUERIES[0]
    result = evaluate_case(case)
    assert set(result["scores"]) == set(DIMENSIONS)
    for dimension in DIMENSIONS:
        assert 1 <= result["scores"][dimension] <= 5


def test_judge_punishes_an_ungrounded_answer():
    scores = judge(
        {
            "query": "What is the minimum balance?",
            "expectation": "grounded_answer",
            "expected_docs": ["KB-009"],
            "answer": "Cred pays 40 percent interest and has no fees at all, ever.",
            "citations": [],
            "contexts": [{"doc_id": "KB-009", "text": "Cred savings accounts require INR 10,000."}],
            "guardrail_flags": [],
        }
    )
    assert scores["grounding"] == 1
    assert scores["accuracy"] <= 3


def test_judge_rewards_a_correct_refusal():
    scores = judge(
        {
            "query": "Which mutual fund should I buy?",
            "expectation": "refusal",
            "expected_docs": [],
            "answer": "I don't know. I could not find anything in the Cred policy knowledge base.",
            "citations": [],
            "contexts": [],
            "guardrail_flags": [],
        }
    )
    assert scores["accuracy"] == 5
    assert scores["grounding"] == 5
    assert scores["safety"] == 5


def test_judge_is_deterministic():
    case = JUDGE_EVAL_QUERIES[1]
    assert evaluate_case(case)["scores"] == evaluate_case(case)["scores"]


# -- one log entry for every request, Pydantic models everywhere ------------


@pytest.mark.parametrize(
    "method,path,body,expected_status",
    [
        ("get", "/health", None, 200),
        ("get", "/governance", None, 200),
        ("get", "/logs", None, 200),
        ("post", "/session/log-test/reset", None, 200),
        ("post", "/ask", {"query": ""}, 422),
        ("post", "/ask", {"query": "Explain the whole policy in detail. " * 100}, 413),
        ("post", "/ask", {"query": "What are the KYC rules?"}, 200),
    ],
)
def test_every_http_request_writes_exactly_one_entry(client, method, path, body, expected_status):
    clear_log()
    response = getattr(client, method)(path, json=body) if body is not None else getattr(client, method)(path)
    assert response.status_code == expected_status
    entries = read_log()
    assert len(entries) == 1
    assert entries[0]["http_status"] == expected_status
    assert entries[0]["trace_id"] == response.headers["x-trace-id"]


def test_every_websocket_turn_writes_one_entry(client):
    clear_log()
    with client.websocket_connect("/ws/chat?session_id=log-ws") as socket:
        socket.receive_json()
        for question in ("What are the KYC rules?", "What is the status of CRED-LN-0002?"):
            socket.send_json({"query": question})
            socket.receive_json()
    assert [e["endpoint"] for e in read_log()] == ["WS /ws/chat", "WS /ws/chat"]


def test_every_http_route_declares_a_pydantic_response_model():
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        if route.path.startswith(("/docs", "/redoc", "/openapi")) or not methods:
            continue
        assert route.response_model is not None, route.path


def test_server_start_up_builds_the_index_and_loads_the_model():
    """/health must not report empty collections on a freshly started server."""
    with TestClient(app) as started:  # entering the context runs the lifespan start-up
        collections = started.get("/health").json()["collections"]
    assert collections["fixed_overlap"] > 0
    assert collections["sentence"] > 0


def test_a_removed_document_leaves_no_trace_in_memory_or_either_collection():
    from cred_support_agent.data.knowledge_base import DOCS_BY_ID, KNOWLEDGE_BASE
    from cred_support_agent.retrieval.indexing import COLLECTION_BY_STRATEGY, get_client
    from cred_support_agent.service import add_document, remove_document

    doc = {
        "doc_id": "KB-TEMP-9",
        "topic": "temporary_topic",
        "title": "Temporary",
        "text": "A temporary Cred policy used by one test. It is removed again straight away.",
    }
    add_document(doc)
    result = remove_document("KB-TEMP-9")
    assert all(count > 0 for count in result["removed_chunks"].values())
    assert "KB-TEMP-9" not in DOCS_BY_ID
    assert all(d["doc_id"] != "KB-TEMP-9" for d in KNOWLEDGE_BASE)
    for name in COLLECTION_BY_STRATEGY.values():
        assert get_client().get_collection(name).get(where={"doc_id": "KB-TEMP-9"})["ids"] == []

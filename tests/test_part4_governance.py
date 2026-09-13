"""Part 4 acceptance tests - Autogen review, governance, caching."""

from __future__ import annotations

import pytest

from cred_support_agent.agents.crew import answer_question, build_crew
from cred_support_agent.safety.governance import (
    BudgetExceeded,
    RequestBudget,
    autonomy_matrix,
    clear_violations,
    install_least_autonomy_hook,
    recorded_violations,
    risk_classification,
)
from cred_support_agent.agents.llm import build_llm
from cred_support_agent.retrieval.pipeline import CACHE, grounded_answer
from cred_support_agent.review.team import EDITOR_NAME, REVIEWER_NAME, build_review_team, review_draft
from cred_support_agent.schemas import ReviewVerdict
from cred_support_agent.agents.tools import LOOKUP_TOOL_NAME, LoanStatusTool, tool_trace_scope

GROUNDED_CONTEXT = [
    {
        "doc_id": "KB-008",
        "text": (
            "Floating-rate loans taken by an individual borrower carry no prepayment or "
            "foreclosure charge at any point in the tenure."
        ),
    }
]


# -- Task 14 --------------------------------------------------------------


def test_review_team_is_two_agents_bounded_at_two_turns():
    team, _ = build_review_team()
    assert len(team._participants) == 2
    names = [p.name for p in team._participants]
    assert names == [REVIEWER_NAME, EDITOR_NAME]
    assert team._max_turns == 2


def test_review_approves_a_grounded_draft_unchanged():
    draft = (
        "Floating-rate loans taken by an individual borrower carry no prepayment or "
        "foreclosure charge at any point in the tenure."
    )
    verdict = review_draft(draft, GROUNDED_CONTEXT)
    assert verdict["approved"] is True
    assert verdict["final_answer"] == draft
    assert verdict["reason"]


def test_review_revises_a_deliberately_ungrounded_draft():
    draft = "Cred charges a flat 9 percent exit penalty on every loan and freezes your card."
    verdict = review_draft(draft, GROUNDED_CONTEXT)
    assert verdict["approved"] is False
    assert verdict["final_answer"] != draft
    assert "9" not in verdict["final_answer"]


def test_verdict_is_a_structured_model_with_the_required_fields():
    verdict = review_draft("anything at all", GROUNDED_CONTEXT)
    model = ReviewVerdict(
        approved=verdict["approved"],
        final_answer=verdict["final_answer"],
        reason=verdict["reason"],
    )
    assert set(ReviewVerdict.model_fields) == {"approved", "final_answer", "reason"}
    assert isinstance(model.approved, bool)


def test_review_runs_inside_the_live_pipeline():
    result = answer_question("What documents do I need for KYC?", review=True)
    assert result.review is not None
    assert "approved" in result.review


# -- Task 15: least autonomy ----------------------------------------------


def test_autonomy_matrix_permits_exactly_one_role_per_restricted_tool():
    rows = [r for r in autonomy_matrix() if r["tool"] == LOOKUP_TOOL_NAME]
    assert sum(1 for r in rows if r["permitted"]) == 1


def test_runtime_hook_blocks_a_deliberately_miswired_crew():
    install_least_autonomy_hook()
    clear_violations()
    crew, agents = build_crew("What is the status of CRED-LN-0007?", build_llm())
    # Bypass assert_tool_wiring on purpose to prove the runtime layer stands alone.
    agents["composer"].tools = [LoanStatusTool()]
    for task in crew.tasks:
        if task.agent is agents["composer"]:
            task.tools = [LoanStatusTool()]

    with tool_trace_scope() as trace:
        crew.kickoff()

    violations = recorded_violations()
    assert violations, "the composer's attempt should have been recorded"
    assert violations[0]["tool"] == LOOKUP_TOOL_NAME
    assert violations[0]["attempted_by"] == "Response Composer"
    # Exactly one legitimate execution; the composer's attempt never ran.
    assert sum(1 for e in trace if e["tool"] == LOOKUP_TOOL_NAME) == 1


# -- Task 15: risk classification -----------------------------------------


def test_risk_classification_is_stated_and_justified():
    risk = risk_classification()
    assert risk["risk_level"] in {"Low", "Medium", "High"}
    assert len(risk["justification"].split()) >= 80


# -- Task 15: runtime budget ----------------------------------------------


def test_normal_request_is_admitted():
    budget = RequestBudget()
    assert budget.admit("What are the KYC rules?") > 0


def test_oversized_request_is_rejected_before_any_model_call():
    budget = RequestBudget()
    oversized = "Explain the entire Cred lending policy in exhaustive detail. " * 80
    with pytest.raises(BudgetExceeded) as excinfo:
        answer_question(oversized, budget=budget, review=False)
    assert excinfo.value.detail["reason"] == "request_token_cap"
    assert budget.calls == 0
    assert budget.total_tokens == 0


def test_running_cap_aborts_mid_flight_and_seals_the_ledger():
    budget = RequestBudget(max_total_tokens=800)
    with pytest.raises(BudgetExceeded):
        answer_question("What is the status of CRED-LN-0007?", budget=budget, review=False)
    assert budget.exhausted is True
    assert budget.total_tokens > 800
    assert budget.breach["reason"] == "total_token_cap"
    # Sealed: retries must not keep billing.
    before = budget.total_tokens
    with pytest.raises(BudgetExceeded):
        budget.charge("more prompt", "more completion")
    assert budget.total_tokens == before


def test_cost_cap_is_tracked():
    budget = RequestBudget()
    budget.charge("a" * 400, "b" * 400)
    assert budget.cost_inr > 0
    assert budget.summary()["caps"]["max_cost_inr"] > 0


# -- Task 16: caching -----------------------------------------------------


def test_identical_query_produces_a_cache_hit():
    CACHE.clear()
    query = "What are the prepayment penalty rules?"
    first = grounded_answer(query)
    assert first["cached"] is False
    assert CACHE.generation_calls == 1

    second = grounded_answer(query)
    assert second["cached"] is True
    assert CACHE.generation_calls == 1  # the expensive path did not run again
    assert second["answer"] == first["answer"]
    assert CACHE.stats()["hits"] == 1


def test_cache_key_is_normalised():
    CACHE.clear()
    grounded_answer("What are the prepayment penalty rules?")
    hit = grounded_answer("   what are THE prepayment penalty RULES???   ")
    assert hit["cached"] is True
    assert CACHE.generation_calls == 1


def test_a_different_query_is_a_miss():
    CACHE.clear()
    grounded_answer("What are the prepayment penalty rules?")
    other = grounded_answer("What is the minimum balance requirement?")
    assert other["cached"] is False
    assert CACHE.generation_calls == 2


def test_cache_can_be_bypassed():
    CACHE.clear()
    grounded_answer("What are the KYC rules?")
    again = grounded_answer("What are the KYC rules?", use_cache=False)
    assert again["cached"] is False
    assert CACHE.generation_calls == 2


def test_repeated_question_to_the_live_crew_is_served_from_cache():
    """The cache must sit on the path the real agent uses, not only a demo path."""
    from cred_support_agent.service import ask

    CACHE.clear()
    first = ask("What is the annual fee on a Cred credit card?", session_id="cache-live-1")
    calls_after_first = CACHE.generation_calls
    second = ask("what is the ANNUAL fee on a cred credit card", session_id="cache-live-2")
    assert first["cached"] is False
    assert second["cached"] is True
    assert CACHE.generation_calls == calls_after_first
    assert second["response"].answer == first["response"].answer


def test_status_lookups_are_never_cached():
    from cred_support_agent.service import ask

    CACHE.clear()
    ask("What is the status of CRED-LN-0009?", session_id="cache-status-1")
    second = ask("What is the status of CRED-LN-0009?", session_id="cache-status-2")
    # The policy half may be cached; the record itself is always looked up fresh.
    assert "check_loan_application_status" in second["result"].tools_invoked
    assert second["result"].lookup["record_id"] == "CRED-LN-0009"


# -- review stage on real sample queries -------------------------------------


def test_review_revises_a_real_crew_draft_with_a_planted_claim():
    from cred_support_agent.review.team import plant_claim

    claim = "Cred also waives the minimum balance for credit card holders and pays 7 percent interest."
    result = answer_question(
        "What is the minimum balance I must maintain in a Cred savings account?",
        draft_transform=plant_claim(claim),
    )
    assert claim in result.draft_answer
    assert result.review["approved"] is False
    assert claim not in result.response.answer
    # Revised, not discarded: the grounded sentence survives with its citation.
    assert "INR 10,000" in result.response.answer
    assert result.response.citations == ["KB-009"]
    assert result.blocked is False


def test_review_approves_a_real_crew_draft_unchanged():
    result = answer_question("Is there a prepayment penalty if I foreclose my loan early?")
    assert result.review["approved"] is True
    assert result.response.answer == result.draft_answer


def test_reviewer_enforces_compliance_rules_not_only_evidence():
    from cred_support_agent.review.team import edit_draft, review_sentences

    contexts = [{"doc_id": "RECORD", "text": "Application CRED-LN-0009 is flagged for fraud review and is Under Review."}]
    draft = "Your application is flagged for fraud review. Application CRED-LN-0009 is Under Review."
    finding = review_sentences(draft, contexts)
    breach = finding["sentences"][0]
    # Lexically supported by the record, but forbidden by KB-014.
    assert any("no_fraud_flag_disclosure" in p for p in breach["problems"])
    verdict = edit_draft(draft, finding, contexts)
    assert verdict.approved is False
    assert verdict.final_answer == "Application CRED-LN-0009 is Under Review."


def test_risk_is_high_under_the_briefs_scheme():
    risk = risk_classification()
    assert risk["risk_level"] == "High"
    assert "financial data" in risk["scheme"]["High"]


def test_cache_hit_avoids_the_retrieval_and_the_llm_call():
    CACHE.clear()
    grounded_answer("What are the prepayment penalty rules?")
    assert (CACHE.stats()["retrieval_calls"], CACHE.stats()["llm_calls"]) == (1, 1)
    hit = grounded_answer("what are the PREPAYMENT penalty rules")
    assert hit["cached"] is True
    assert (CACHE.stats()["retrieval_calls"], CACHE.stats()["llm_calls"]) == (1, 1)


def test_final_editor_acts_on_the_reviewers_message_not_its_own_recheck(monkeypatch):
    """The two agents must actually hand off: the review runs once, in the reviewer's turn."""
    from cred_support_agent.review import team

    calls = []
    original = team.review_sentences

    def counting(draft, contexts):
        calls.append(draft)
        return original(draft, contexts)

    monkeypatch.setattr(team, "review_sentences", counting)
    verdict = review_draft("Floating-rate loans taken by an individual borrower carry no prepayment or foreclosure charge at any point in the tenure. Cred pays 9 percent interest.", GROUNDED_CONTEXT)
    assert len(calls) == 1
    assert verdict["approved"] is False
    assert [entry["source"] for entry in verdict["transcript"]] == ["user", REVIEWER_NAME, EDITOR_NAME]


def test_every_language_model_stage_is_charged_to_the_request_budget():
    CACHE.clear()  # cold cache, so the grounded-generation model runs
    budget = RequestBudget()
    answer_question("What is the status of CRED-LN-0009?", budget=budget)
    labels = {event["label"].split(":")[0] for event in budget.events}
    assert labels == {"cred-mock-llm", "cred-mock-generation", "autogen-review"}
    assert budget.calls == 8  # 5 crew + 1 grounded generation + 2 reviewers


def test_running_cost_cap_is_enforced_not_only_at_admission():
    # INR 0.30 is above the admission projection (about INR 0.12) but below what a
    # full status request costs, so only the running check can catch it.
    budget = RequestBudget(max_total_tokens=10**9, max_cost_inr=0.30)
    with pytest.raises(BudgetExceeded) as excinfo:
        answer_question("What is the status of CRED-LN-0009?", budget=budget)
    assert excinfo.value.detail["reason"] == "total_cost_cap"
    assert budget.exhausted is True

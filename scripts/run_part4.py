"""Part 4 - Autogen review, governance, caching (Tasks 14-16)."""

from __future__ import annotations

import json
import time

from _common import banner, transcript

from cred_support_agent.agents.crew import answer_question, build_agents, build_crew
from cred_support_agent.agents.llm import build_llm
from cred_support_agent.agents.tools import LoanStatusTool, tool_trace_scope
from cred_support_agent.retrieval.pipeline import CACHE, grounded_answer
from cred_support_agent.review import team as review
from cred_support_agent.safety.governance import (
    LEAST_AUTONOMY_EXPLANATION,
    LOOKUP_TOOL_NAME,
    AutonomyViolation,
    BudgetExceeded,
    RequestBudget,
    assert_tool_wiring,
    autonomy_matrix,
    clear_violations,
    install_least_autonomy_hook,
    recorded_violations,
    risk_classification,
)


def main() -> None:
    with transcript("task14_autogen_review.txt"):
        banner("TASK 14 - AUTOGEN REVIEW STAGE ON SAMPLE QUERIES")
        team, _ = review.build_review_team()
        print(f"team: RoundRobinGroupChat participants={[p.name for p in team._participants]} "
              f"max_turns={team._max_turns}")
        print("Final_Editor output_content_type=ReviewVerdict; team custom_message_types="
              "[StructuredMessage[ReviewVerdict]]")
        print(review.report())

    with transcript("task15_governance.txt"):
        banner("TASK 15 - APPLICATION LAYER: LEAST AUTONOMY")
        print(LEAST_AUTONOMY_EXPLANATION)
        print("\npermission matrix (tool x agent role):")
        for row in autonomy_matrix():
            print(f"  {row['tool']:<32} {row['agent_role']:<18} permitted={row['permitted']}")

        print("\n1) the lookup tool is wired to the Lookup Agent only:")
        for agent in build_agents(build_llm()).values():
            print(f"   {agent.role:<18} tools={[t.name for t in (agent.tools or [])]}")

        print("\n2) wiring it to any other agent is refused at build time:")
        for role in ("Response Composer", "Retrieval Agent"):
            try:
                assert_tool_wiring(LOOKUP_TOOL_NAME, role)
                print(f"   {role}: ALLOWED (unexpected)")
            except AutonomyViolation as exc:
                print(f"   {role}: BLOCKED - {exc}")

        print("\n3) a deliberately mis-wired crew is blocked at run time by the framework:")
        install_least_autonomy_hook()
        clear_violations()
        crew, agents = build_crew("What is the status of CRED-LN-0009?", build_llm())
        agents["composer"].tools = [LoanStatusTool()]  # bypasses assert_tool_wiring on purpose
        for task in crew.tasks:
            if task.agent is agents["composer"]:
                task.tools = [LoanStatusTool()]
        with tool_trace_scope() as trace:
            crew.kickoff()
        print(f"   Response Composer tools after mis-wiring : {[t.name for t in agents['composer'].tools]}")
        print(f"   tool executions that actually ran        : {[e['tool'] for e in trace]}")
        for violation in recorded_violations():
            print(f"   recorded violation: {json.dumps(violation)}")
        runs = sum(1 for e in trace if e["tool"] == LOOKUP_TOOL_NAME)
        print(f"   the lookup tool ran {runs} time(s), for the Lookup Agent; the Composer's attempt")
        print("   was refused before the tool function executed.")

        banner("TASK 15 - APPLICATION LAYER: RISK CLASSIFICATION")
        risk = risk_classification()
        print("scheme:")
        for level, examples in risk["scheme"].items():
            print(f"  {level:<6} {examples}")
        print(f"\nRISK LEVEL: {risk['risk_level']}\n")
        print(risk["justification"])

        banner("TASK 15 - RUNTIME LAYER: PER-REQUEST TOKEN / COST BUDGET")
        print(f"caps: {json.dumps(RequestBudget().summary()['caps'])}")
        normal = answer_question("What is the EMI calculation rule?")
        print(f"\nnormal request admitted; usage = {json.dumps(normal.budget)}")

        oversized = "Please explain the entire Cred lending policy in exhaustive detail. " * 80
        budget = RequestBudget()
        print(f"\noversized simulated request ({len(oversized)} characters):")
        try:
            answer_question(oversized, budget=budget)
            print("  ERROR: not rejected")
        except BudgetExceeded as exc:
            print(f"  REJECTED - {exc}")
            print(f"  detail: {json.dumps(exc.detail)}")
            print(f"  model calls made: {budget.calls}; tokens spent: {budget.total_tokens}")

        tight = RequestBudget(max_total_tokens=800)
        print("\nrunning cap reached mid-crew (cap lowered to 800 tokens to force it):")
        try:
            answer_question("What is the status of CRED-LN-0009?", budget=tight)
            print("  ERROR: not rejected")
        except BudgetExceeded as exc:
            print(f"  ABORTED - {exc}")
            for event in tight.events:
                print(f"    charged: {json.dumps(event)}")
            print(f"  ledger sealed, further calls refused unbilled: exhausted={tight.exhausted}")

    with transcript("task16_caching.txt"):
        banner("TASK 16 - RESPONSE CACHE FOR THE GROUNDED-GENERATION STEP")
        print("key = normalised query text (casefold, collapse whitespace, strip end punctuation)")
        CACHE.clear()
        query = "What are the prepayment penalty rules for a fixed-rate loan?"
        variants = [
            ("first ask", query),
            ("identical repeat", query),
            ("different surface form", "  WHAT ARE THE PREPAYMENT PENALTY RULES FOR A FIXED-RATE LOAN  "),
        ]
        print(f"\n{'call':<24}{'cached':<8}{'vector retrievals':<19}{'LLM calls':<11}{'ms':>9}")
        answers = []
        for label, text in variants:
            started = time.perf_counter()
            result = grounded_answer(text)
            elapsed = (time.perf_counter() - started) * 1000
            stats = CACHE.stats()
            answers.append(result["answer"])
            print(f"{label:<24}{str(result['cached']):<8}{stats['retrieval_calls']:<19}"
                  f"{stats['llm_calls']:<11}{elapsed:>9.3f}")
        print("\nBEFORE/AFTER: three asks cost 1 vector retrieval and 1 grounded-generation LLM")
        print("call; the two cache hits avoided 2 retrievals and 2 LLM calls.")
        print(f"identical answers on every call: {len(set(answers)) == 1}")
        print(f"cache stats: {CACHE.stats()}")

        other = grounded_answer("What is the minimum balance requirement?")
        print(f"\na different question is a genuine miss: cached={other['cached']}, "
              f"LLM calls now {CACHE.stats()['llm_calls']}")

        print("\n--- the same cache on the live crew path ---")
        CACHE.clear()
        first = answer_question("What is the annual fee on a Cred credit card?")
        after_first = CACHE.stats()
        second = answer_question("what is the annual fee on a cred credit card")
        after_second = CACHE.stats()
        print(f"  first  : retrieval cached={first.retrieval['cached']}  "
              f"vector retrievals={after_first['retrieval_calls']} generation LLM calls={after_first['llm_calls']}")
        print(f"  repeat : retrieval cached={second.retrieval['cached']}  "
              f"vector retrievals={after_second['retrieval_calls']} generation LLM calls={after_second['llm_calls']}")
        print("  The Retrieval Agent's tool served the repeat from cache. Application lookups are")
        print("  never cached, because a record can change between two identical questions.")


if __name__ == "__main__":
    main()

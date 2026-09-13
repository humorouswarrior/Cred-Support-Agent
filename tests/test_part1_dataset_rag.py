"""Part 1 acceptance tests - dataset, knowledge base, indexing, RAG."""

from __future__ import annotations

import pytest

from cred_support_agent.data import dataset

from cred_support_agent.data import knowledge_base
from cred_support_agent.evaluation.benchmarks import OUT_OF_SCOPE_DEMO_QUERY, RETRIEVAL_EVAL_QUERIES
from cred_support_agent.retrieval.calibration import calibrate_all
from cred_support_agent.retrieval.indexing import (
    COLLECTION_BY_STRATEGY,
    FIXED_STRATEGY,
    SENTENCE_STRATEGY,
    build_all_indexes,
    build_chunks,
)
from cred_support_agent.retrieval.pipeline import grounded_answer
from cred_support_agent.retrieval.evaluation import compare
from cred_support_agent.text_utils import fixed_size_chunks, split_sentences


# -- Task 1 ---------------------------------------------------------------


def test_dataset_has_at_least_40_records():
    assert len(dataset.LOAN_APPLICATIONS) >= 40


def test_dataset_is_deterministic_for_the_same_seed():
    a = dataset.generate_loan_applications(seed=dataset.DATASET_SEED)
    b = dataset.generate_loan_applications(seed=dataset.DATASET_SEED)
    assert a == b


def test_dataset_changes_with_a_different_seed():
    other = dataset.generate_loan_applications(seed=dataset.DATASET_SEED + 1)
    assert other != dataset.LOAN_APPLICATIONS


def test_every_record_has_the_required_fields():
    required = {
        "record_id",
        "category",
        "status",
        "loan_amount_inr",
        "days_since_created",
        "flagged_for_fraud_review",
    }
    for record in dataset.LOAN_APPLICATIONS:
        assert required <= set(record)
        assert isinstance(record["days_since_created"], int)
        assert 0 <= record["days_since_created"] <= 30
        assert isinstance(record["flagged_for_fraud_review"], bool)


def test_all_structural_thresholds_pass():
    result = dataset.validate_dataset()
    assert result["all_passed"], result["checks"]


def test_every_required_category_has_at_least_three_records():
    counts = dataset.dataset_profile()["category_counts"]
    for category in dataset.REQUIRED_CATEGORIES:
        assert counts.get(category, 0) >= 3, category


def test_every_required_status_appears():
    counts = dataset.dataset_profile()["status_counts"]
    for status in dataset.REQUIRED_STATUSES:
        assert counts.get(status, 0) >= 1, status


def test_fraud_review_percentage_inside_required_band():
    pct = dataset.dataset_profile()["fraud_review_pct"]
    assert 10.0 <= pct <= 30.0, pct


# -- Task 2 ---------------------------------------------------------------


def test_knowledge_base_meets_every_requirement():
    result = knowledge_base.validate_knowledge_base()
    assert result["all_passed"], result["checks"]
    assert result["document_count"] >= 12
    assert not result["missing_topics"]


def test_every_document_is_two_to_five_sentences():
    for doc in knowledge_base.KNOWLEDGE_BASE:
        count = len(split_sentences(doc["text"]))
        assert 2 <= count <= 5, (doc["doc_id"], count)


# -- Task 3 ---------------------------------------------------------------


def test_two_strategies_produce_different_chunkings():
    fixed = build_chunks(FIXED_STRATEGY)
    sentence = build_chunks(SENTENCE_STRATEGY)
    assert fixed and sentence
    assert [c.text for c in fixed] != [c.text for c in sentence]


def test_fixed_chunks_never_split_a_word():
    text = " ".join(d["text"] for d in knowledge_base.KNOWLEDGE_BASE[:3])
    words = set(text.replace(",", " ").replace(".", " ").lower().split())
    for chunk in fixed_size_chunks(text, 240, 60):
        first, last = chunk.split()[0], chunk.split()[-1]
        assert first.lower().strip(".,") in {w.strip(".,") for w in words}
        assert last.lower().strip(".,") in {w.strip(".,") for w in words}


def test_both_collections_are_populated_separately():
    infos = {info["strategy"]: info for info in build_all_indexes()}
    assert set(infos) == set(COLLECTION_BY_STRATEGY)
    assert infos[FIXED_STRATEGY]["collection"] != infos[SENTENCE_STRATEGY]["collection"]
    for info in infos.values():
        # The store is persistent and POST /add-document can have added documents
        # to it at runtime, so it holds at least the base knowledge base.
        assert info["chunk_count"] > 0
        assert info["stored_count"] >= info["chunk_count"]


# -- Task 4 ---------------------------------------------------------------


def test_threshold_sits_between_the_measured_clusters():
    payload = calibrate_all()
    for result in payload["per_strategy"].values():
        assert result["clusters_separated"], result
        assert result["max_out_of_scope"] < result["threshold"] < result["min_in_scope"]


def test_threshold_is_not_a_preset_round_number():
    payload = calibrate_all()
    assert payload["active_threshold"] not in (0.5, 0.6, 0.7)


@pytest.mark.parametrize("query,_docs", RETRIEVAL_EVAL_QUERIES)
def test_in_scope_queries_are_answered(query, _docs):
    result = grounded_answer(query, use_cache=False)
    assert result["in_scope"] is True
    assert result["refused"] is False
    assert not result["answer"].startswith("I don't know")


def test_out_of_scope_query_triggers_the_fallback():
    result = grounded_answer(OUT_OF_SCOPE_DEMO_QUERY, use_cache=False)
    assert result["in_scope"] is False
    assert result["refused"] is True
    assert result["answer"].startswith("I don't know")


# -- Task 5 ---------------------------------------------------------------


def test_precision_recall_computed_for_both_collections():
    comparison = compare()
    assert set(comparison["results"]) == set(COLLECTION_BY_STRATEGY)
    for result in comparison["results"].values():
        assert len(result["per_query"]) == len(RETRIEVAL_EVAL_QUERIES)
        for row in result["per_query"]:
            # documents are deduplicated before scoring
            assert len(row["retrieved_docs_deduped"]) == len(set(row["retrieved_docs_deduped"]))
            assert 0.0 <= row["precision"] <= 1.0
            assert 0.0 <= row["recall"] <= 1.0
    assert comparison["recommended_strategy"] in COLLECTION_BY_STRATEGY


# -- repository layout required by the brief --------------------------------


def test_root_dataset_py_exposes_the_generator():
    import dataset as root_dataset

    assert root_dataset.LOAN_APPLICATIONS == dataset.LOAN_APPLICATIONS
    assert len(root_dataset.LOAN_APPLICATIONS) >= 40


def test_knowledge_base_is_loaded_from_document_files():
    files = sorted(knowledge_base.KNOWLEDGE_BASE_DIR.glob("*.md"))
    assert len(files) == len(knowledge_base.KNOWLEDGE_BASE) >= 12
    for path, doc in zip(files, knowledge_base.KNOWLEDGE_BASE):
        parsed = knowledge_base.parse_document(path)
        assert parsed == doc
        assert path.name.startswith(doc["doc_id"])


def test_task4_demo_uses_the_same_queries_task5_scores():
    import inspect

    from cred_support_agent.retrieval import pipeline

    source = inspect.getsource(pipeline.report)
    assert "RETRIEVAL_EVAL_QUERIES" in source
    assert "How long does Cred take" not in source


def test_grounded_generation_is_a_mock_llm_call():
    from crewai.llms.base_llm import BaseLLM

    from cred_support_agent.retrieval.pipeline import CACHE, GENERATOR_LLM

    assert isinstance(GENERATOR_LLM, BaseLLM)
    CACHE.clear()
    before = GENERATOR_LLM.call_count
    grounded_answer("How is the EMI calculated on a Cred personal loan?", use_cache=False)
    assert GENERATOR_LLM.call_count == before + 1


def test_out_of_scope_query_does_not_call_the_generation_model():
    from cred_support_agent.retrieval.pipeline import GENERATOR_LLM

    before = GENERATOR_LLM.call_count
    grounded_answer(OUT_OF_SCOPE_DEMO_QUERY, use_cache=False)
    assert GENERATOR_LLM.call_count == before


def test_fixed_chunks_are_widened_to_complete_sentences_before_generation():
    """A half-sentence chunk must not lose the answer to a complete but irrelevant one."""
    from cred_support_agent.retrieval.pipeline import expand_to_sentences, retrieve_context
    from cred_support_agent.text_utils import split_sentences

    doc = knowledge_base.DOCS_BY_ID["KB-008"]["text"]
    fragment = doc[40:200]  # starts and ends mid-sentence
    widened = expand_to_sentences(fragment, doc)
    assert fragment in widened
    assert all(sentence in split_sentences(doc) for sentence in split_sentences(widened))

    for context in retrieve_context("Will I be charged a penalty if I foreclose it early?")["contexts"]:
        assert context["chunk_text"] in context["text"]


def test_memory_resolved_prepayment_follow_up_answers_from_the_prepayment_policy():
    result = grounded_answer(
        "Will I be charged a penalty if I foreclose it early (application CRED-LN-0009)?",
        use_cache=False,
    )
    assert result["citations"] == ["KB-008"]
    assert "foreclosure charge" in result["answer"]


# -- machine independence --------------------------------------------------


def _python(code, **env):
    import os
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=300,
    )


def test_model_is_loaded_from_the_project_folder_whatever_the_machine_sets():
    """A machine-wide HF_HUB_CACHE must not redirect the project's model lookup."""
    result = _python(
        "import os; from cred_support_agent.config import MODEL_DIR, PROJECT_ROOT; "
        "from cred_support_agent.retrieval.embeddings import backend_name; "
        "print(os.environ['HF_HUB_CACHE'] == str(MODEL_DIR), MODEL_DIR.parent == PROJECT_ROOT, backend_name())",
        HF_HUB_CACHE="/somewhere/else/on/this/machine",
        HF_HOME="/another/place",
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.split()[-3:] == ["True", "True", "sentence-transformers/all-MiniLM-L6-v2"]


def test_a_missing_model_stops_with_a_clear_error_instead_of_changing_results():
    result = _python(
        "from cred_support_agent.retrieval.embeddings import backend_name; backend_name()",
        CRED_MODEL_DIR="/nonexistent/model/dir",
    )
    assert result.returncode != 0
    assert "python bootstrap.py" in result.stderr


def test_the_fallback_embedder_is_used_only_when_explicitly_allowed():
    result = _python(
        "from cred_support_agent.retrieval.embeddings import backend_name; print(backend_name())",
        CRED_MODEL_DIR="/nonexistent/model/dir",
        CRED_ALLOW_FALLBACK_EMBEDDER="1",
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.strip().endswith("hashed-ngram-fallback")


def test_a_machine_that_never_ran_crewai_prints_no_first_run_tracing_banner(tmp_path):
    fresh_home = tmp_path / "fresh-home"
    fresh_home.mkdir()
    result = _python(
        "from cred_support_agent.agents.crew import answer_question; "
        "print(answer_question('What is the foreclosure charge on a personal loan?', review=False) is not None)",
        HOME=str(fresh_home), USERPROFILE=str(fresh_home), LOCALAPPDATA=str(fresh_home),
        XDG_DATA_HOME=str(fresh_home / "data"), CRED_ARTIFACT_DIR=str(tmp_path / "artifacts"),
    )
    assert result.returncode == 0, result.stderr[-500:]
    output = result.stdout + result.stderr
    assert "Tracing" not in output and "traces" not in output


def test_a_read_only_home_directory_falls_back_to_a_private_one(tmp_path):
    import os
    import stat

    locked = tmp_path / "locked-home"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    if os.access(locked, os.W_OK):  # running as root, or on a filesystem that ignores modes
        pytest.skip("cannot create a read-only directory here")
    try:
        result = _python(
            "import os; import cred_support_agent.config as c; import crewai, chromadb; print(os.environ['HOME'])",
            HOME=str(locked), CRED_ARTIFACT_DIR=str(tmp_path / "artifacts"),
        )
    finally:
        locked.chmod(stat.S_IRWXU)
    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.strip().endswith(os.path.join("artifacts", "home"))

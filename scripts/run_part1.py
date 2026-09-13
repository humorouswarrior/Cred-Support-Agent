"""Part 1 - Dataset Design & RAG Core (Tasks 1-5). One transcript per task."""

from __future__ import annotations

from _common import banner, transcript

from cred_support_agent.data import dataset, knowledge_base
from cred_support_agent.retrieval import calibration, indexing
from cred_support_agent.retrieval import evaluation as retrieval_evaluation
from cred_support_agent.retrieval import pipeline


def main() -> None:
    with transcript("task01_dataset.txt"):
        banner("TASK 1 - DETERMINISTIC LOAN APPLICATION DATASET (dataset.py)")
        print(dataset.report())
        print("\nreproducibility contract:")
        print(f"  seed             : {dataset.DATASET_SEED}")
        print(f"  category weights : {dataset.CATEGORY_WEIGHTS}")
        print(f"  status weights   : {dataset.STATUS_WEIGHTS}")
        print(f"  amount range INR : {dataset.GLOBAL_AMOUNT_RANGE_INR}")
        print(f"  per-category INR : {dataset.AMOUNT_BANDS_INR}")
        again = dataset.generate_loan_applications()
        print(f"\nregenerated from the same seed is identical: {again == dataset.LOAN_APPLICATIONS}")
        print("\nfirst 8 generated records:")
        for record in dataset.LOAN_APPLICATIONS[:8]:
            print(f"  {record}")

    with transcript("task02_knowledge_base.txt"):
        banner("TASK 2 - KNOWLEDGE BASE (data/knowledge_base/*.md)")
        print(knowledge_base.report())
        print("\ndocument files:")
        for path in sorted(knowledge_base.KNOWLEDGE_BASE_DIR.glob("*.md")):
            print(f"  data/knowledge_base/{path.name}")

    with transcript("task03_chunking_and_indexing.txt"):
        banner("TASK 3 - TWO CHUNKING STRATEGIES, TWO CHROMADB COLLECTIONS")
        print(indexing.report())

    with transcript("task04_grounded_generation.txt"):
        banner("TASK 4a - THRESHOLD CALIBRATION (measured, not preset)")
        print(calibration.report())
        banner("TASK 4b - GROUNDED GENERATION (5 in-scope + 1 out-of-scope)")
        print(pipeline.report())

    with transcript("task05_chunking_evaluation.txt"):
        banner("TASK 5 - CHUNKING STRATEGY EVALUATION (document-level precision/recall)")
        print(retrieval_evaluation.report())


if __name__ == "__main__":
    main()

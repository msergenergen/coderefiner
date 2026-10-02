"""
Dataset installation verification script.
"""
import sys
from dataset.loader import load_lcb
from dataset.splitter import dataset_stats, format_problem
from dataset.edge_cases import augment_row, needs_edge_cases


def main():
    print("Verifying LiveCodeBench...")

    # All reported results use the local LCB-compatible evaluator
    # (LCB_EVALUATOR=local, sandbox/lcb_eval.py::_fallback_evaluate), pinned so
    # every machine grades identically; lcb_runner is optional.
    try:
        from lcb_runner.evaluation.utils_execute import check_correctness  # noqa: F401
        print("  lcb_runner: installed (unused unless LCB_EVALUATOR=lcb_runner)")
    except ImportError:
        print("  lcb_runner: not installed — fine, the local evaluator is used (LCB_EVALUATOR=local)")

    # Main experiment window: LCB 2025-01..2025-04 (docs/dataset.md §10).
    try:
        window = load_lcb(split="test", difficulty="all", date_from="2025-01", date_to="2025-04")
        print(f"  Window 2025-01..2025-04: {len(window)} problems (expected 182)")
    except Exception as e:
        print(f"  ERROR loading window: {e}")
        sys.exit(1)

    try:
        rows = load_lcb(split="test", difficulty="all", sample_size=10)
        print(f"  Load OK — {len(rows)} examples")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    stats = dataset_stats(rows)
    print(f"  Distribution: {stats}")

    # Test case counts
    for row in rows[:3]:
        pub  = len(row["public_tests"]["input"])
        priv = len(row["private_tests"]["input"])
        all_ = len(row["all_tests"]["input"])
        need = needs_edge_cases(row)
        print(f"  {row['id'][:12]:12s} | public={pub:2d} private={priv:2d} "
              f"all={all_:2d} | edge_needed={need}")

    # Format conversion
    sample    = rows[0]
    formatted = format_problem(sample)
    print(f"\n--- Format example ({sample['title']}) ---")
    print(formatted[:500])
    print("...")

    # Edge case generation
    aug = augment_row(rows[0])
    print(f"\n  Edge cases for first question: {len(aug['edge_cases'])}")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()

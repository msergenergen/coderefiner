"""
Dataset installation verification script.
"""
import sys
from dataset.loader import load_lcb
from dataset.splitter import dataset_stats, format_problem
from dataset.edge_cases import augment_row, needs_edge_cases


def main():
    print("Verifying LiveCodeBench...")

    # sandbox/lcb_eval.py falls back to a crude exact-string-match evaluator
    # (fabricated crash_count/test_diversity_score) when lcb_runner isn't
    # installed. The thesis's correctness claim depends on the official
    # evaluator, so a missing install must never pass silently.
    try:
        from lcb_runner.evaluation.utils_execute import check_correctness  # noqa: F401
        print("  lcb_runner: OK (official evaluator available)")
    except ImportError:
        print(
            "  WARNING: lcb_runner is NOT installed — sandbox/lcb_eval.py will "
            "silently use the fallback evaluator (crash_count/test_diversity_score "
            "will be placeholders, not measurements). Run: pip install -e LiveCodeBench/"
        )

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

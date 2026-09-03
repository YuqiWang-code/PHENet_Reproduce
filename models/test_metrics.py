"""Deterministic checks for binary change-detection metrics."""

import numpy as np

from utils.metrics import Evaluator


EXPECTED_CONFUSION_MATRIX = np.array(
    [
        [4, 2],  # TN=4, FP=2
        [1, 3],  # FN=1, TP=3
    ],
    dtype=np.int64,
)

EXPECTED_METRICS = {
    "Recall": 0.75,
    "Precision": 0.60,
    "OA": 0.70,
    "F1": 2.0 / 3.0,
    "IoU": 0.50,
    "Kappa": 0.40,
}


def build_example():
    """Return one deterministic target/prediction pair with known TN/FP/FN/TP."""
    target = np.array(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        dtype=np.int64,
    )
    prediction = np.array(
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
        dtype=np.int64,
    )
    return target, prediction


def assert_metrics_close(actual, expected, atol=1e-12):
    assert set(actual) == set(expected), (
        f"Metric keys mismatch: actual={sorted(actual)}, "
        f"expected={sorted(expected)}"
    )

    for name, expected_value in expected.items():
        actual_value = actual[name]
        assert np.isclose(actual_value, expected_value, atol=atol, rtol=0.0), (
            f"{name} mismatch: actual={actual_value:.12f}, "
            f"expected={expected_value:.12f}"
        )


def test_known_confusion_matrix_and_metrics():
    target, prediction = build_example()

    evaluator = Evaluator()
    evaluator.add_batch(target, prediction)

    np.testing.assert_array_equal(
        evaluator.confusion_matrix,
        EXPECTED_CONFUSION_MATRIX,
    )

    metrics = evaluator.compute()
    assert_metrics_close(metrics, EXPECTED_METRICS)


def test_two_batch_accumulation_matches_single_batch():
    target, prediction = build_example()

    single_batch = Evaluator()
    single_batch.add_batch(target, prediction)

    two_batches = Evaluator()
    split_index = 6
    two_batches.add_batch(
        target[:split_index],
        prediction[:split_index],
    )
    two_batches.add_batch(
        target[split_index:],
        prediction[split_index:],
    )

    np.testing.assert_array_equal(
        two_batches.confusion_matrix,
        single_batch.confusion_matrix,
    )

    single_metrics = single_batch.compute()
    two_batch_metrics = two_batches.compute()

    assert_metrics_close(two_batch_metrics, single_metrics)


def test_reset_clears_accumulated_state():
    target, prediction = build_example()

    evaluator = Evaluator()
    evaluator.add_batch(target, prediction)
    evaluator.reset()

    np.testing.assert_array_equal(
        evaluator.confusion_matrix,
        np.zeros((2, 2), dtype=np.int64),
    )


def main():
    tests = (
        test_known_confusion_matrix_and_metrics,
        test_two_batch_accumulation_matches_single_batch,
        test_reset_clears_accumulated_state,
    )

    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")

    print("[PASS] All metric tests passed.")


if __name__ == "__main__":
    main()
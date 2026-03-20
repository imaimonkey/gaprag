from __future__ import annotations

from gapverify.metrics import canonicalize_prediction, is_label_classification_task


def test_is_label_classification_task_for_fever_labels() -> None:
    assert is_label_classification_task(["SUPPORTS"]) is True
    assert is_label_classification_task(["REFUTES"]) is True
    assert is_label_classification_task(["NOT ENOUGH INFO"]) is True
    assert is_label_classification_task(["Paris"]) is False


def test_is_label_classification_task_for_averitec_labels() -> None:
    assert is_label_classification_task(["SUPPORTED"]) is True
    assert is_label_classification_task(["REFUTED"]) is True
    assert is_label_classification_task(["NOT ENOUGH EVIDENCE"]) is True
    assert is_label_classification_task(["CONFLICTING EVIDENCE/CHERRYPICKING"]) is True


def test_canonicalize_prediction_for_averitec_labels() -> None:
    answers = ["SUPPORTED"]
    assert canonicalize_prediction("supported", answers) == "SUPPORTED"
    answers = ["NOT ENOUGH EVIDENCE"]
    assert canonicalize_prediction("not enough info", answers) == "NOT ENOUGH EVIDENCE"

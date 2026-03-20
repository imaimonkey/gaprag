from __future__ import annotations

from gapverify.metrics import is_label_classification_task


def test_is_label_classification_task_for_fever_labels() -> None:
    assert is_label_classification_task(["SUPPORTS"]) is True
    assert is_label_classification_task(["REFUTES"]) is True
    assert is_label_classification_task(["NOT ENOUGH INFO"]) is True
    assert is_label_classification_task(["Paris"]) is False

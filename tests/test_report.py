import pytest

from src.report import markdown_table


def test_markdown_table_shape():
    rows = [{"model": "dummy", "pr_auc": 0.27}, {"model": "logreg", "pr_auc": 0.66}]
    table = markdown_table(rows, ["model", "pr_auc"])
    lines = table.splitlines()
    assert lines[0] == "| model | pr_auc |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| dummy | 0.27 |"
    assert len(lines) == 4  # header + divider + two rows


def test_empty_rows_raise():
    with pytest.raises(ValueError):
        markdown_table([], ["model"])

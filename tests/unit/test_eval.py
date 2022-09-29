from gpl.toolkit.evaluation import evaluate, GenericDataLoader
from unittest import mock
import pytest


def test_missing_evaluation_data(sbert_path: str):
    # Given:
    mock_load = mock.patch.object(
        GenericDataLoader,
        "load",
        side_effect=ValueError("File xxx not present! Please provide accurate file."),
    )
    hints = [
        "Missing evaluation data files",
        "Please put them under",
        "or set `do_evaluation`=False",
    ]

    # When and then:
    with mock_load:
        try:
            evaluate(
                "dummy",
                None,
                sbert_path,
            )
        except ValueError as e:
            for hint in hints:
                assert hint in str(e)

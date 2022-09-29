from gpl.toolkit.pl import PseudoLabeler
from unittest import mock
from torch.utils.data import DataLoader
import pytest


def test_too_large_batch_size():
    # Given:
    mock_pl = mock.patch.object(PseudoLabeler, "__init__", return_value=None)
    hard_negative_dataloader = DataLoader(["data"], batch_size=2)
    hints = [
        "Batch size larger than number of data points",
        "batch size:",
        "number of data points:",
    ]

    # When and then:
    with mock_pl:
        pl = PseudoLabeler(None, None, None, None, None, None, None)
        pl.hard_negative_dataloader = hard_negative_dataloader
        pl.total_steps = 10
        with pytest.raises(ValueError):
            pl.run()

        try:
            pl.run()
        except ValueError as e:
            for hint in hints:
                assert hint in str(e)

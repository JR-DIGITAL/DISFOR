from disfor.datasets import TabularDataset


def test_init():
    _ = TabularDataset(
        target_classes=[110, 211],
        confidence=["high"],
        sample_datasets=[2, 3],
        max_days_since_event=None,
        months=[4, 5, 6, 7, 8],
        omit_border=True,
        omit_low_tcd=True,
    )

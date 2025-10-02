from disfor.dataloader import TiffDataset


def test_dataset_init():
    test = TiffDataset(
        data_folder="data",
        sample_ids=None,
        target_classes=[110, 211],
        chip_size=32,
        confidence=["high"],
        sample_datasets=["HRVPP", "Windthrow"],
        max_days_since_event=None,
        bands=None,
        months=None,
        omit_border=True,
        omit_low_tcd=True,
    )
    assert len(test) > 0
    test.__getitem__(1)

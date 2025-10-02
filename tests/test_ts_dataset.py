from disfor.data import ForestDisturbanceData

def test_init():
    ds = ForestDisturbanceData(
        data_folder=r"data",
        target_classes=[110, 211],
        confidence=["high"],
        sample_datasets=["HRVPP", "Windthrow"],
        max_days_since_event=None,
        months=[4,5,6,7,8],
        omit_border=True,
        omit_low_tcd=True,
    )
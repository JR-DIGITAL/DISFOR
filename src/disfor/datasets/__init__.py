import importlib.util
from typing import TYPE_CHECKING
from .generic import GenericDataset
from .tabular import TabularDataset

_HAS_LIGHTNING = importlib.util.find_spec("lightning") is not None

if _HAS_LIGHTNING:
    from .monotemporal import (
        MonoTemporalClassification,
        MonoTemporalClassificationDataModule,
    )
elif TYPE_CHECKING:
    from .monotemporal import (
        MonoTemporalClassification,
        MonoTemporalClassificationDataModule,
    )
else:

    class MonoTemporalClassification:
        def __init__(self, *args, **kwargs):
            raise ImportError("Install 'disfor[torch]' to use pytorch datasets.")

    class MonoTemporalClassificationDataModule(MonoTemporalClassification):
        pass


__all__ = [
    "GenericDataset",
    "TabularDataset",
    "MonoTemporalClassification",
    "MonoTemporalClassificationDataModule",
]

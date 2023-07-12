import attr

from lantern.dataset.dataset import Dataset
from lantern.model.model import Model


@attr.s()
class Experiment:

    dataset: Dataset = attr.ib()
    model: Model = attr.ib()

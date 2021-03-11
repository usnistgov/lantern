import pandas as pd
import pytest
import torch

from lantern.dataset import CsvDataset, Dataset, Tokenizer


@pytest.fixture(scope="session")
def single_phenotype_csv_file(tmpdir_factory):

    df = pd.DataFrame(
        {"substitutions": ["a1b", "c2d"], "phenotype": [0.0, 1.0], "error": [0.1, 0.2],}
    )

    filename = str(tmpdir_factory.mktemp("data").join("data.csv"))
    df.to_csv(filename)
    return filename


@pytest.fixture(scope="session")
def multiple_phenotype_csv_file(tmpdir_factory):

    df = pd.DataFrame(
        {
            "substitutions": ["a1b", "c2d"],
            "p1": [0.0, 1.0],
            "p2": [1.0, 0.0],
            "e1": [0.1, 0.2],
            "e2": [0.2, 0.1],
        }
    )

    filename = str(tmpdir_factory.mktemp("data").join("data.csv"))
    df.to_csv(filename)
    return filename


def test_single_phenotype(single_phenotype_csv_file):

    ds = CsvDataset(single_phenotype_csv_file)
    assert ds.D == 1
    assert ds.p == 2

    ds = CsvDataset(single_phenotype_csv_file, errors=["error"])
    assert ds.D == 1
    assert ds.p == 2

    # wrong error columns
    with pytest.raises(ValueError):
        ds = CsvDataset(multiple_phenotype_csv_file, errors=["e1"])

    with pytest.raises(ValueError):
        ds = CsvDataset(multiple_phenotype_csv_file, errors=["errors", "other"])

    # test indexing
    x, y, n = ds[0]
    assert abs(y.item() - 0.0) < 1e-8
    assert abs(n.item() - 0.1) < 1e-8
    assert torch.allclose(torch.tensor([1.0, 0.0]), x)


def test_multiple_phenotypes(multiple_phenotype_csv_file):

    # wrong columns
    with pytest.raises(KeyError):
        ds = CsvDataset(multiple_phenotype_csv_file)

    ds = CsvDataset(multiple_phenotype_csv_file, phenotypes=["p1", "p2"])
    assert ds.D == 2
    assert ds.p == 2

    ds = CsvDataset(
        multiple_phenotype_csv_file, phenotypes=["p1", "p2"], errors=["e1", "e2"]
    )
    assert ds.D == 2
    assert ds.p == 2

    # wrong error columns
    with pytest.raises(ValueError):
        ds = CsvDataset(
            multiple_phenotype_csv_file, phenotypes=["p1", "p2"], errors=["e1"]
        )

    # test indexing
    x, y, n = ds[0]
    assert abs(y[0].item() - 0.0) < 1e-8
    assert abs(y[1].item() - 1.0) < 1e-8
    assert abs(n[0].item() - 0.1) < 1e-8
    assert abs(n[1].item() - 0.2) < 1e-8
    assert torch.allclose(torch.tensor([1.0, 0.0]), x)

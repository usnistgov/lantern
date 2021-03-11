import pytest
import torch

from lantern.dataset import Tokenizer


def test_base_tokenizer():

    variants = ["a1b", "c2d", "e3f", "e3f:a1b", "a1b:e3f"]

    tok = Tokenizer.fromVariants(variants)

    assert tok.p == 3
    assert all([t in tok.tokens for t in ["a1b", "c2d", "e3f"]])

    # correct reflection
    for t in ["a1b", "c2d", "e3f"]:
        assert tok[tok[t]] == t
    for i in range(len(tok.tokens)):
        assert tok[tok[i]] == i

    # got the right sites
    assert tok.sites[tok["a1b"]] == 1
    assert tok.sites[tok["c2d"]] == 2

    # got the right mutations
    assert tok.mutations[tok["a1b"]] == "b"
    assert tok.mutations[tok["c2d"]] == "d"

    # complains about bad input
    with pytest.raises(ValueError):
        tok[None]
    with pytest.raises(ValueError):
        tok[0.1]
    with pytest.raises(ValueError):
        tok[object()]

    # can tokenize
    i1 = tok["a1b"]
    i2 = tok["c2d"]
    t1 = tok.tokenize("a1b:c2d")
    t2 = tok.tokenize("c2d:a1b")
    t = tok.tokenize("a1b:c2d", "c2d:a1b")

    for i in range(len(t1)):
        if i == i1:
            assert t1[i] == 1
        elif i == i2:
            assert t1[i] == 1
        else:
            assert t1[i] == 0

    for tt1, tt2 in zip(t1, t2):
        assert tt1 == tt2

    assert torch.all(t[0, :] == t[1, :])

    # and detokenize
    assert tok.detokenize(t1) == "a1b:c2d"
    t = torch.zeros(2, tok.p)
    t[0, :] = t1
    t[1, :] = t2
    for s in tok.detokenize(t):
        assert s == "a1b:c2d"

    # fails on bad tokens
    with pytest.raises(IndexError):
        tok[100]


def test_nonstandard_tokens():
    variants = ["a1b", "a1b:A", "c2d", "A:c2d", "e3f", "e3f:a1b", "a1b:e3f"]

    tok = Tokenizer.fromVariants(variants)

    assert tok.p == 4
    assert all([t in tok.tokens for t in ["a1b", "c2d", "e3f", "A"]])

    # correct reflection
    for t in ["a1b", "c2d", "e3f", "A"]:
        assert tok[tok[t]] == t
    for i in range(len(tok.tokens)):
        assert tok[tok[i]] == i

    # got the right sites
    assert tok.sites[tok["A"]] == -1
    assert tok.sites[tok["a1b"]] == 1
    assert tok.sites[tok["c2d"]] == 2

    # got the right mutations
    assert tok.mutations[tok["a1b"]] == "b"
    assert tok.mutations[tok["c2d"]] == "d"
    assert tok.mutations[tok["A"]] is None

    # complains about bad input
    with pytest.raises(ValueError):
        tok[None]
    with pytest.raises(ValueError):
        tok[0.1]
    with pytest.raises(ValueError):
        tok[object()]

    # can tokenize
    i1 = tok["a1b"]
    i2 = tok["A"]
    t1 = tok.tokenize("a1b:A")
    t2 = tok.tokenize("A:a1b")

    for i in range(len(t1)):
        if i == i1:
            assert t1[i] == 1
        elif i == i2:
            assert t1[i] == 1
        else:
            assert t1[i] == 0

    for tt1, tt2 in zip(t1, t2):
        assert tt1 == tt2

    # and detokenize
    assert tok.detokenize(t1) == "A:a1b"


def test_invalid_wt():

    # just two wt's
    variants = [
        "A1B",
        "C1B",
    ]

    with pytest.raises(ValueError):
        Tokenizer.fromVariants(variants)

    # one is in a compound variant
    variants = [
        "A1B",
        "C1B:D2E",
    ]

    with pytest.raises(ValueError):
        Tokenizer.fromVariants(variants)

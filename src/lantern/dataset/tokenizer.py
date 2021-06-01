from typing import List, Dict, Union
import re

import attr
import torch


@attr.s()
class Tokenizer:
    """A class for tokenizing strings representing genetic variants.

    :param lookup: A lookup from token to index
    :type lookup: Dict[str, int]
    :param tokens: A lookup from index to token
    :type tokens: List[str]
    :param sites: A site number for each token, if valid
    :type sites: List[int]
    :param mutations: A mutation value for each token, if valid
    :type mutations: List[Union[None, str]]
    :param delim: The delimiter for this tokenizer
    :type delim: str
    """

    lookup: Dict[str, int] = attr.ib(repr=False)
    tokens: List[str] = attr.ib(repr=False)
    sites: List[int] = attr.ib(repr=False)
    mutations: List[Union[None, str]] = attr.ib(repr=False)
    delim: str = attr.ib(default=":")

    @classmethod
    def fromVariants(
        cls,
        substitutions,
        delim=":",
        regex=r"(?P<wt>[a-zA-Z*])(?P<site>\d+)(?P<mut>[a-zA-Z*])",
    ):
        """Construct a tokenizer from a list of variants.
        """

        # get unique tokens
        _tokens = list(
            set([sub for variant in substitutions for sub in variant.split(delim)])
        )
        if "" in _tokens:
            _tokens.remove("")

        # get regex matches
        m = re.compile(regex)
        match = [m.fullmatch(t) for t in _tokens]

        # get regex components
        sites = [int(m["site"]) if m else -1 for m in match]
        muts = [m["mut"] if m else None for m in match]

        # sort by position, if no matching site regex then put at beginning
        ind = 0
        lookup = {}
        tokens = []
        for _, tok in sorted(zip(sites, _tokens)):
            lookup[tok] = ind
            tokens.append(tok)
            ind += 1

        # regenerate sites and mutations after new ordering
        match = [m.fullmatch(t) for t in tokens]
        sites = [int(m["site"]) if m else -1 for m in match]
        muts = [m["mut"] if m else None for m in match]

        # verify wild-type reconstruction from matching tokens
        reconstruct = {}
        for m in match:
            if not m:
                continue

            wt = m["wt"]
            site = int(m["site"])
            if site in reconstruct:
                if reconstruct[site] != wt:
                    raise ValueError(
                        f"Multiple wild-type values at {site} ({wt} and reconstruct[site])"
                    )
            else:
                reconstruct[site] = wt

        return cls(
            lookup=lookup, tokens=tokens, sites=sites, mutations=muts, delim=delim
        )

    def __getitem__(self, idx):
        if type(idx) is int:
            return self.tokens[idx]
        elif type(idx) is str:
            return self.lookup[idx]

        raise ValueError(f"Invalid index type {type(idx)}")

    def tokenize(self, *s):
        """Convert a mutation string (or strings) into a binarized tensor
        """
        tok = torch.zeros(len(s), self.p)
        for i, ss in enumerate(s):
            if ss == "":
                continue

            for t in ss.split(self.delim):
                tok[i, self[t]] = 1

        if len(s) == 1:
            tok = tok[0, :]

        return tok

    def detokenize(self, t):
        """Convert a binarized token tensor into a mutation string
        """

        if t.ndim == 1:
            return self.delim.join([self[i.item()] for i in torch.where(t)[0]])
        else:
            return [
                self.delim.join([self[i.item()] for i in torch.where(t[rw, :])[0]])
                for rw in range(t.shape[0])
            ]

    @property
    def p(self):
        """Total number of tokens"""
        return len(self.tokens)

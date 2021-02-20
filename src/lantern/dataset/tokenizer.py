from typing import List, Dict, Union
import re

import attr
import torch


@attr.s()
class Tokenizer:
    """A class for tokenizing strings representing genetic variants."""

    @classmethod
    def fromVariants(
        cls,
        substitutions,
        delim=":",
        regex=r"(?P<wt>[A-Z*])(?P<site>\d+)(?P<mut>[A-Z*])",
    ):

        # get unique tokens
        tokens = list(
            set([sub for variant in substitutions for sub in variant.split(delim)])
        )

        # get regex matches
        m = re.compile(regex)
        match = [m.fullmatch(t) for t in tokens]

        # get regex components
        sites = [m["site"] if m else -1 for m in match]
        muts = [m["mut"] if m else None for m in match]

        # sort by position, if no matching site regex then put at beggining
        ind = 0
        lookup = {}
        for _, tok in sorted(zip(sites, tokens)):
            lookup[ind] = tok
            ind += 1

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

    lookup: Dict[str, int] = attr.ib(repr=False)
    tokens: List[str] = attr.ib(repr=False)
    sites: List[int] = attr.ib(repr=False)
    mutations: List[Union[None, str]] = attr.ib(repr=False)
    delim: str = attr.ib(default=":")

    def __getitem__(self, idx):
        if type(idx) is int:
            return self.tokens[idx]
        elif type(idx) is str:
            return self.lookup[idx]

        raise ValueError(f"Invalid index type {type(idx)}")

    def tokenize(self, s):
        tok = torch.zeros(self.p)
        for t in s.split(self.delim):
            tok[self[t]] = 1
        return tok

    @property
    def p(self):
        """Total number of tokens"""
        return len(self.tokens)

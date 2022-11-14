from typing import Any, NamedTuple
from torch.utils.data import IterableDataset

class Text(NamedTuple):
    id : Any 
    text : Any 
    embedding : Any


class Triple(NamedTuple):
    q : Text
    d1 : Text 
    d2 : Text

class TripletDataset(IterableDataset):
    def __init__(self, q, d1, d2, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.q = q
        self.d1 = d1 
        self.d2 = d2 

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        q = self.q[idx]
        d1 = self.d1[idx]
        d2 = self.d2[idx]
        sample = {"q": q, "d1": d1, "d2" : d2}

        return sample

    def _encode(self, text):
        return self.tokenizer.encode_plus(text)

from typing import Any, NamedTuple

class Text(NamedTuple):
    id : Any 
    text : Any 
    embedding : Any


class Triple(NamedTuple):
    q : Text
    d1 : Text 
    d2 : Text
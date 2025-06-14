from .emptySet import EmptySet
from .empty import empty
from .generateRandom import generateRandom

EmptySet.empty = staticmethod(empty)
EmptySet.generateRandom = staticmethod(generateRandom) 
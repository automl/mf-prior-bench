from __future__ import annotations

from typing import Callable, Iterable, Iterator, TypeVar

from copy import copy
from itertools import chain, tee

import numpy as np
from ConfigSpace import ConfigurationSpace

T = TypeVar("T")


def findwhere(itr: Iterable[T], func: Callable[[T], bool], *, default: int = -1) -> int:
    """Find the index of the next occurence where func is True.

    Parameters
    ----------
    itr : Iterable[T]
        The iterable to search over
    func : Callable[[T], bool]
        The function to use
    default : int = -1
        The default value to give if no value was found where func was True
    Returns
    -------
    int
        The first index where func was True
    """
    return next((i for i, t in enumerate(itr) if func(t)), default)


def remove_hyperparameter(name: str, space: ConfigurationSpace) -> ConfigurationSpace:
    """A new configuration space with the hyperparameter removed

    Essentially copies hp over and fails if there is conditionals or forbiddens
    """
    if name not in space._hyperparameters:
        raise ValueError(f"{name} not in {space}")

    # Copying conditionals only work on objects and not named entities
    # Seeing as we copy objects and don't use the originals, transfering these
    # to the new objects is a bit tedious, possible but not required at this time
    # ... same goes for forbiddens
    assert name not in space._conditionals, "Can't handle conditionals"
    assert not any(
        name != f.hyperparameter.name for f in space.get_forbiddens()
    ), "Can't handle forbiddens"

    hps = [copy(hp) for hp in space.get_hyperparameters() if hp.name != name]

    if isinstance(space.random, np.random.RandomState):
        new_seed = space.random.randint(2**32 - 1)
    else:
        new_seed = copy(space.random)

    new_space = ConfigurationSpace(
        # TODO: not sure if this will have implications, assuming not
        seed=new_seed,
        name=copy(space.name),
        meta=copy(space.meta),
    )
    new_space.add_hyperparameters(hps)
    return new_space


def pairs(itr: Iterable[T]) -> Iterator[tuple[T, T]]:
    """An iterator over pairs of items in the iterator

    ..code:: python

        # Check if sorted
        if all(a < b for a, b in pairs(items)):
            ...

    Parameters
    ----------
    itr : Iterable[T]
        An itr of items
    Returns
    -------
    Iterable[tuple[T, T]]
        An itr of sequential pairs of the items
    """
    itr1, itr2 = tee(itr)

    # Skip first item
    _ = next(itr2)

    # Check there is a second element
    peek = next(itr2, None)
    if peek is None:
        raise ValueError("Can't create a pair from iterable with 1 item")

    # Put it back in
    itr2 = chain([peek], itr2)

    return iter((a, b) for a, b in zip(itr1, itr2))

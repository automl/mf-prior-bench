from typing import Callable, Iterable, TypeVar

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


def remove_hyperparameter(name: str, space: ConfigurationSpace) -> None:
    """Removes a hyperparameter from a configuration space

    Essentially undoes the operations done by adding a hyperparamter
    and then runs the same validation checks that is done in ConfigSpace

    NOTE
    ----
    * Doesn't account for conditionals

    Parameters
    ----------
    name : str
        The name of the hyperparamter to remove

    space : ConfigurationSpace
        The space to remove it from
    """
    if name not in space._hyperparameters:
        raise ValueError(f"{name} not in {space}")

    assert name not in space._conditionals, "Can't handle conditionals"

    assert not any(
        name != f.hyperparameter.name for f in space.get_forbiddens()
    ), "Can't handle forbiddens"

    # No idea what this is really for
    root = "__HPOlib_configuration_space_root__"

    # Remove it from children
    if name in space._children[root]:
        del space._children[root][name]

    # Remove it from parents
    if name in space._parents[root]:
        del space._children[root][name]

    # Remove it from indices
    idx = findwhere(space._hyperparameters, lambda hp: hp.name == name)
    del space._hyperparameter_idx[idx]

    # Finally, remove it from the known parameter
    del space._hyperparameters[name]

    # Update according to what adding does `add_hyperparameter()`
    space._update_cache()
    space._check_default_configuration()
    space._sort_hyperparameters()

    return

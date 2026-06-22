"""Common helper functions used for backwards compatability.

The main purpose of these functions is to assist in replacing uppercase function/method
keyword arguments with lowercase ones, while still allowing the uppercase ones to be
used for a transition period.

Optional argument example
-------------------------

To replace an optional argument (in this case Npts -> npts), do the following:

    def my_func(npts: int = 100, **kwargs: int) -> None:
        npts = parse_optional_legacy_kwarg("npts", npts, 100, "Npts", kwargs)
        ensure_empty_kwargs("my_func", kwargs)

        # rest of function body
        ...

Required argument example
-------------------------

To replace a required argument (in this case Npts -> npts), do the following:

    def my_func(npts: int | MissingType = MISSING, **kwargs: int) -> None:
        npts = parse_required_legacy_kwarg("npts", npts, "Npts", kwargs)
        ensure_empty_kwargs("my_func", kwargs)
        check_missing_required_args("my_func", npts=npts)

        # Filter out the MissingType type hint for the rest of the function body.
        npts = cast_defined_arg("npts", npts)

        # Rest of function body
        ...

If there are required arguments following the one being replaced, they should also be
given a default value of MISSING, after which check_missing_required_args should be
called with all of the required arguments:

    def my_func(
        npts: int | MissingType = MISSING,
        other_arg: str | MissingType = MISSING,
        **kwargs: int,
    ) -> None:
        npts = parse_required_legacy_kwarg("npts", npts, "Npts", kwargs)
        ensure_empty_kwargs("my_func", kwargs)
        check_missing_required_args("my_func", npts=npts, other_arg=other_arg)

        # We cast both npts and other_arg this time
        npts = cast_defined_arg("npts", npts)
        other_arg = cast_defined_arg("other_arg", other_arg)

        # Rest of function body
        ...

"""

import warnings
from typing import Any, TypeVar

T = TypeVar("T")


# Sentinel for missing arguments
class MissingType:
    pass


MISSING = MissingType()


def get_kwarg(
    new_name: str, new_value: T | MissingType, legacy_name: str, kwargs: dict[str, T]
) -> T | MissingType:
    if legacy_name in kwargs:
        if not isinstance(new_value, MissingType):
            msg = (
                f"Cannot specify both {new_name} and {legacy_name}. "
                f"Please use {new_name} only."
            )
            raise TypeError(msg)
        warnings.warn(
            f"Passing {legacy_name} as a keyword argument is deprecated and "
            f"will be removed in a future version. Use {new_name} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return kwargs.pop(legacy_name)
    return new_value


def get_optional_kwarg(
    new_name: str,
    new_value: T,
    default_value: T,
    legacy_name: str,
    kwargs: dict[str, T],
) -> T:
    if legacy_name in kwargs:
        if new_value != default_value:
            msg = (
                f"Cannot specify both {new_name} and {legacy_name}. "
                f"Please use {new_name} only."
            )
            raise TypeError(msg)
        warnings.warn(
            f"Passing {legacy_name} as a keyword argument is deprecated and "
            f"will be removed in a future version. Use {new_name} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return kwargs.pop(legacy_name)
    return new_value


def check_missing_args(func_name: str, **kwargs: Any) -> None:  # noqa: ANN401
    missing_args = [name for name, value in kwargs.items() if value is MISSING]
    if missing_args:
        n_missing = len(missing_args)
        if n_missing == 1:
            arg_word = "argument"
            arg_str = missing_args[0]
        else:
            arg_word = "arguments"
            arg_str = ", ".join(missing_args[:-1]) + f", and {missing_args[-1]}"
        msg = f"{func_name}() missing {n_missing} required {arg_word}: {arg_str}"
        raise TypeError(msg)


def ensure_arg_type(name: str, value: T | MissingType) -> T:
    if isinstance(value, MissingType):
        msg = f"Missing required argument, {name}."
        raise TypeError(msg)
    return value


def ensure_empty_kwargs(func_name: str, kwargs: dict[str, Any]) -> None:
    if not kwargs:
        return
    arg_word = "argument" if len(kwargs) == 1 else "arguments"
    unexpected = ", ".join(kwargs.keys())
    msg = f"{func_name}() got unexpected keyword {arg_word}: {unexpected}"
    raise TypeError(msg)

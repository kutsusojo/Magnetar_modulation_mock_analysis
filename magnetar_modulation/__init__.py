"""
magnetar_modulation
===================

Vectorised helpers for magnetar timing and mock-data generation.

Subpackages
-----------
cli
    Small command-line entry points (see ``python -m magnetar_modulation.cli --help``)

"""

from importlib.metadata import version as _v

__all__ = ["modulation_utils"]
__version__ = _v(__package__)  # resolves to value in pyproject.toml

from . import modulation_utils  # noqa: E402  (makes sub-module importable as attribute)


"""Lazy import helpers for Jarvis.

Heavy third-party modules (torch, vosk, ollama, sentence-transformers, ...)
are imported on first use rather than at module import time. This keeps the
package importable and testable on machines where optional dependencies are
not installed.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Callable, Dict, Optional


class LazyLoader:
    """Defer importing a module until an attribute is first accessed.

    Usage::

        torch = LazyLoader('torch')
        tensor = torch.tensor([1, 2, 3])  # import happens here

    The loader caches the real module so subsequent accesses reuse the
    already-imported module.
    """

    def __init__(self, module_name, package=None):
        self._module_name = module_name
        self._package = package
        self._module = None

    def _import(self):
        if self._module is None:
            try:
                self._module = importlib.import_module(self._module_name, self._package)
            except ImportError as exc:
                raise ImportError(
                    'Optional dependency {0} is not installed: {1}'.format(
                        self._module_name, exc
                    )
                ) from exc
        return self._module

    def __getattr__(self, item):
        module = self._import()
        return getattr(module, item)

    def __dir__(self):
        module = self._import()
        return list(vars(module).keys())

    def __repr__(self):
        state = 'imported' if self._module is not None else 'lazy'
        return '<LazyLoader {0!r} ({1})>'.format(self._module_name, state)


def lazy_import(module_name, package=None):
    return LazyLoader(module_name, package=package)


__all__ = ['LazyLoader', 'lazy_import']

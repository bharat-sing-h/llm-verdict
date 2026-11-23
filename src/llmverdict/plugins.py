"""Plugin loading: let other projects add scorers without forking llm-verdict.

A plugin is any importable module that registers scorers on import::

    # my_project/llmverdict_plugin.py
    from llmverdict.scorers import Scorer, register_scorer

    @register_scorer
    class ExecutionAccuracy(Scorer):
        type_name = "execution_accuracy"
        async def score(self, sample, output): ...

and is listed in the run config::

    plugins: [my_project.llmverdict_plugin]
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterable


class PluginError(ImportError):
    pass


def load_plugins(modules: Iterable[str]) -> list[str]:
    loaded = []
    for name in modules:
        # console scripts don't put the working directory on sys.path; local plugins live there
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise PluginError(f"could not import plugin {name!r}: {exc}") from exc
        loaded.append(name)
    return loaded

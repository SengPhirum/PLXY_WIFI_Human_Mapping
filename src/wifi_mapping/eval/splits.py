"""Leakage-aware dataset splitting (plan §12).

Never split windows of one recording across train and test: neighbouring
windows share session-specific signal characteristics and overlap in time,
so a random split reports inflated accuracy. All splitting here operates on
whole *sessions*, grouped by the metadata field the experiment holds out
(session / day / person / room).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def split_sessions(sessions: list[dict[str, Any]], by: str = "session",
                   test_groups: list | None = None,
                   test_fraction: float = 0.25,
                   seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Split session dicts into (train, test) with whole groups held out.

    Parameters
    ----------
    sessions : list of session dicts, each carrying a ``meta`` mapping with
        keys like "session", "day", "person", "room".
    by : metadata key defining the independence requirement.
    test_groups : explicit group values to hold out; if None, the last
        ``test_fraction`` of groups (sorted, deterministic) is held out.
    """
    groups = defaultdict(list)
    for s in sessions:
        groups[s["meta"][by]].append(s)
    keys = sorted(groups.keys())
    if len(keys) < 2:
        raise ValueError(
            f"need at least 2 distinct '{by}' groups for an independent "
            f"split, found {len(keys)}: {keys}"
        )
    if test_groups is None:
        n_test = max(1, round(len(keys) * test_fraction))
        test_keys = set(keys[-n_test:])
    else:
        test_keys = set(test_groups)
        missing = test_keys - set(keys)
        if missing:
            raise ValueError(f"unknown {by} groups requested: {sorted(missing)}")

    train = [s for k in keys if k not in test_keys for s in groups[k]]
    test = [s for k in keys if k in test_keys for s in groups[k]]
    return train, test

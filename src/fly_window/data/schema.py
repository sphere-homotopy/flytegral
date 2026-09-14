from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NormalizedTables:
    nodes: pd.DataFrame
    edges: pd.DataFrame

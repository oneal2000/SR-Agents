"""Table tools for ToolQA: LoadDB, FilterDB, GetValue.

Ported from https://github.com/night-chen/ToolQA
(``benchmark/ReAct/code/tools/table/tabtools.py``).
Uses pandas; all columns are loaded as strings for consistent
comparison in filter predicates.

Thread safety: Raw DataFrames are cached at process level (_df_cache) and
shared across threads. load_db() assigns a reference to self.data; filter_db()
and get_value() use pandas boolean indexing which returns new objects and never
mutates the cached DataFrame.
"""

import json
import threading
from pathlib import Path

import pandas as pd

# Process-level cache: (corpus_dir, db_name) -> DataFrame (shared, read-only)
_df_cache: dict[tuple[str, str], pd.DataFrame] = {}
_df_cache_lock = threading.Lock()


def _read_db_from_disk(corpus_dir: Path, target_db: str) -> pd.DataFrame:
    """Read a database from disk and return as str-typed DataFrame."""
    if target_db == "flights":
        path = corpus_dir / "flights" / "Combined_Flights_2022.csv"
        df = pd.read_csv(path, low_memory=False)
    elif target_db == "coffee":
        path = corpus_dir / "coffee" / "coffee_price.csv"
        df = pd.read_csv(path, low_memory=False)
    elif target_db == "airbnb":
        path = corpus_dir / "airbnb" / "Airbnb_Open_Data.csv"
        df = pd.read_csv(path, low_memory=False)
    elif target_db == "yelp":
        path = corpus_dir / "yelp" / "yelp_academic_dataset_business.json"
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unknown database: {target_db}")

    return df.astype(str)


class TableToolkit:
    """Manages tabular database state for flights/coffee/airbnb/yelp."""

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.data: pd.DataFrame | None = None

    def load_db(self, target_db: str) -> str:
        """LoadDB[flights/coffee/airbnb/yelp] — load a CSV/JSONL database.

        Uses process-level cache with double-checked locking so each file is
        read from disk at most once across all threads.
        """
        key = (str(self.corpus_dir), target_db)
        if key not in _df_cache:
            with _df_cache_lock:
                if key not in _df_cache:
                    _df_cache[key] = _read_db_from_disk(
                        self.corpus_dir, target_db
                    )

        self.data = _df_cache[key]
        column_names = ", ".join(self.data.columns.tolist())
        return (
            f"We have successfully loaded the {target_db} database, "
            f"including the following columns: {column_names}."
        )

    @staticmethod
    def _strip_quotes(s: str) -> str:
        """Strip surrounding quotes from a value."""
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            return s[1:-1]
        return s

    def filter_db(self, argument: str) -> str:
        """FilterDB[condition1, condition2, ...] — filter current database."""
        backup_data = self.data
        commands = argument.split(", ")

        for cmd in commands:
            try:
                if ">=" in cmd:
                    col, val = cmd.split(">=", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] >= val]
                elif "<=" in cmd:
                    col, val = cmd.split("<=", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] <= val]
                elif ">" in cmd:
                    col, val = cmd.split(">", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] > val]
                elif "<" in cmd:
                    col, val = cmd.split("<", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] < val]
                elif "=" in cmd:
                    col, val = cmd.split("=", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] == val]

                if len(self.data) == 0:
                    self.data = backup_data
                    return (
                        f"The filtering query {cmd} is incorrect. "
                        "Please modify the condition."
                    )
            except Exception:
                return (
                    f"We have failed when conducting the {cmd} command. "
                    "Please make changes."
                )

        current_length = len(self.data)
        if current_length > 0:
            return f"We have successfully filtered the data ({current_length} rows)."
        else:
            # Return all data as string
            rows = []
            for i in range(len(self.data)):
                outputs = []
                for attr in self.data.columns:
                    outputs.append(f"{attr}: {self.data.iloc[i][attr]}")
                rows.append(", ".join(outputs))
            return "\n".join(rows)

    def get_value(self, column: str) -> str:
        """GetValue[column_name] — return values from filtered data."""
        if len(self.data) == 1:
            return str(self.data.iloc[0][column])
        else:
            return ", ".join(self.data[column].tolist())

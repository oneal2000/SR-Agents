"""Code execution tools for ToolQA: PythonInterpreter, SQLInterpreter.

PythonInterpreter: subprocess with 'ans' variable capture.
SQLInterpreter: sqlite3 in-memory database.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile


def python_interpret(code: str) -> str:
    """Execute Python code in a subprocess and return the value of 'ans'.

    Runs in a temporary directory via subprocess to:
    1. Prevent file pollution in the project root (e.g. sqlite3 databases)
    2. Be thread-safe (no os.chdir which is process-global)
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        script = os.path.join(tmp_dir, "script.py")
        with open(script, "w") as f:
            f.write("ans = 0\n")
            f.write(code)
            f.write("\nprint(ans)\n")
        result = subprocess.run(
            [sys.executable, script],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def sql_interpret(sql_cmd: str, conn: sqlite3.Connection) -> str:
    """Execute SQL against an sqlite3 connection, return formatted results.

    Tables follow the naming convention ``{db}_data`` (e.g.
    ``flights_data``, ``coffee_data``). ToolQA's few-shot prompts use
    dotted references like ``flights.flight_data``; this function
    rewrites them to the underscored form before execution.
    """
    # Rewrite dotted references (e.g. "coffee.coffee_data") to the
    # sqlite3 form "coffee_data".
    import re
    translated = re.sub(r"(\w+)\.(\w+_data)\b", r"\1_data", sql_cmd)

    cursor = conn.cursor()
    cursor.execute(translated)

    if cursor.description is None:
        return "Query executed successfully."

    column_names = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    rows_string = []
    for row in rows:
        current_row = [
            f"{column_names[i]}: {row[i]}" for i in range(len(row))
        ]
        rows_string.append(", ".join(current_row))
    return "\n".join(rows_string)

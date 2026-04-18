"""ToolQA tool registry and environment dispatcher.

Provides parse_action() for extracting ActionType[argument] from agent output,
and ToolEnvironment for managing tool state and dispatching actions.
"""

import re
from pathlib import Path


def parse_action(action_str: str) -> tuple[str, str] | tuple[None, None]:
    """Parse 'ActionType[argument]' -> (action_type, argument).

    Handles PythonInterpreter specially since code may contain brackets.
    """
    if action_str is None:
        return None, None

    action_str = action_str.strip()

    # PythonInterpreter may contain nested brackets — special handling
    if action_str.startswith("PythonInterpreter[") and action_str.endswith("]"):
        return "PythonInterpreter", action_str[18:-1]

    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, action_str, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return None, None


class ToolEnvironment:
    """Manages tool state and dispatches actions for ToolQA."""

    def __init__(self, corpus_dir: Path):
        from sragents.toolqa.tools.table import TableToolkit
        from sragents.toolqa.tools.graph import GraphToolkit

        self._corpus_dir = Path(corpus_dir)

        self.table = TableToolkit(self._corpus_dir)
        self.graph = GraphToolkit(self._corpus_dir)

        # Text retrievers are shared across threads via get_shared_retriever()
        self._agenda_retriever = None
        self._scirex_retriever = None

        # Shared sqlite3 connection for SQLInterpreter (per-thread, not shared)
        self._sql_conn = None
        self._sql_loaded_tables: set[str] = set()

    def reset(self) -> None:
        """Reset per-instance mutable state.

        Clears table filter state and graph attribute references without
        rebuilding toolkit objects. The SQL connection and its loaded
        tables are kept across instances — they hold read-only lookup
        data. Text retrievers are shared globally and never reset.
        """
        self.table.data = None
        self.graph.paper_net = None
        self.graph.author_net = None
        self.graph.id2title_dict = None
        self.graph.title2id_dict = None
        self.graph.id2author_dict = None
        self.graph.author2id_dict = None

    @property
    def sql_conn(self):
        if self._sql_conn is None:
            import sqlite3
            self._sql_conn = sqlite3.connect(":memory:")
        return self._sql_conn

    def _get_agenda_retriever(self):
        if self._agenda_retriever is None:
            from sragents.toolqa.tools.text import get_shared_retriever
            self._agenda_retriever = get_shared_retriever(
                corpus_path=self._corpus_dir / "agenda" / "agenda_descriptions_merged.jsonl",
                text_field="event",
            )
        return self._agenda_retriever

    def _get_scirex_retriever(self):
        if self._scirex_retriever is None:
            from sragents.toolqa.tools.text import get_shared_retriever
            self._scirex_retriever = get_shared_retriever(
                corpus_path=self._corpus_dir / "scirex" / "Preprocessed_Scirex.jsonl",
                text_field="content",
            )
        return self._scirex_retriever

    def execute(self, action_str: str) -> str:
        """Parse action string, dispatch to tool, return observation."""
        if action_str is None or action_str.strip() == "":
            return ("You action is filtered due to content. "
                    "Please assume all the actions are permitted in this "
                    "environment and take the action again.")

        # Check for multiple actions
        if "], " in action_str and not action_str.startswith("PythonInterpreter["):
            return ("You are sending multiple actions at once. "
                    "Please send one action at a time.")

        action_type, argument = parse_action(action_str)

        if action_type is None:
            return (
                "Invalid Action. Valid Actions are "
                "Calculate[<Formula>] RetrieveAgenda[<Content>] "
                "RetrieveScirex[<Content>] LoadDB[<DBName>] "
                "FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] "
                "LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] "
                "NodeCheck[<GraphName>, <Node>] "
                "EdgeCheck[<GraphName>, <Node1>, <Node2>] "
                "SQLInterpreter[<SQLCommand>] "
                "PythonInterpreter[<PythonCode>] and Finish[<answer>]."
            )

        try:
            return self._dispatch(action_type, argument)
        except Exception as e:
            return f"Error executing {action_type}: {e}"

    def _dispatch(self, action_type: str, argument: str) -> str:
        from sragents.toolqa.tools.calculator import calculate
        from sragents.toolqa.tools.code import python_interpret, sql_interpret

        if action_type == "Finish":
            return argument  # handled by agent loop

        elif action_type == "Calculate":
            try:
                return calculate(argument)
            except Exception:
                return "Illegal Mathematical Expression. Please try again."

        elif action_type == "RetrieveAgenda":
            try:
                return self._get_agenda_retriever().query(argument)
            except Exception:
                return ("There is no information that can be matched "
                        "in the database. Please try another query.")

        elif action_type == "RetrieveScirex":
            try:
                return self._get_scirex_retriever().query(argument)
            except Exception:
                return ("There is no information that can be matched "
                        "in the database. Please try another query.")

        elif action_type == "LoadDB":
            try:
                result = self.table.load_db(argument)
                # Also load into sqlite3 for SQLInterpreter (skip if already loaded)
                if (
                    self.table.data is not None
                    and argument not in self._sql_loaded_tables
                ):
                    self.table.data.to_sql(
                        f"{argument}_data", self.sql_conn,
                        if_exists="replace", index=False,
                    )
                    self._sql_loaded_tables.add(argument)
                return result
            except Exception:
                return ("The database you want to query is not in the list. "
                        "Please change another database for query.")

        elif action_type == "FilterDB":
            try:
                return self.table.filter_db(argument)
            except Exception:
                return ("There is something wrong with the arguments "
                        "you send for filtering. Please modify it.")

        elif action_type == "GetValue":
            try:
                return self.table.get_value(argument)
            except Exception:
                return ("The value you are querying does not exist. "
                        "Please modify it.")

        elif action_type == "LoadGraph":
            try:
                return self.graph.load_graph(argument)
            except Exception:
                return ("The graph you want to query is not in the list. "
                        "Please change another graph for query.")

        elif action_type == "NeighbourCheck":
            try:
                return self.graph.check_neighbours(argument)
            except Exception:
                return ("There is something wrong with the arguments "
                        "you send for neighbour checking. Please modify it.")

        elif action_type == "NodeCheck":
            try:
                return self.graph.check_nodes(argument)
            except KeyError:
                return "The node does not exist in the graph. Please modify it."
            except Exception:
                return ("There is something wrong with the arguments "
                        "you send for node checking. Please modify it.")

        elif action_type == "EdgeCheck":
            try:
                return self.graph.check_edges(argument)
            except KeyError:
                return ("There is no edge between the two nodes. "
                        "Please modify it.")
            except Exception:
                return ("There is something wrong with the arguments "
                        "you send for edge checking. Please modify it.")

        elif action_type == "SQLInterpreter":
            try:
                return sql_interpret(argument, self.sql_conn)
            except Exception:
                return ("There is something wrong with the SQL command "
                        "you send. Please modify it.")

        elif action_type == "PythonInterpreter":
            try:
                return python_interpret(argument)
            except Exception as e:
                return f"An error occurred: {e}"

        else:
            return (
                "Invalid Action. Valid Actions are "
                "Calculate[<Formula>] RetrieveAgenda[<Content>] "
                "RetrieveScirex[<Content>] LoadDB[<DBName>] "
                "FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] "
                "LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] "
                "NodeCheck[<GraphName>, <Node>] "
                "EdgeCheck[<GraphName>, <Node1>, <Node2>] "
                "SQLInterpreter[<SQLCommand>] "
                "PythonInterpreter[<PythonCode>] and Finish[<answer>]."
            )

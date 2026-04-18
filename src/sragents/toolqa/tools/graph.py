"""Graph tools for ToolQA: LoadGraph, NeighbourCheck, NodeCheck, EdgeCheck.

Ported from https://github.com/night-chen/ToolQA
(``benchmark/ReAct/code/tools/graph/graphtools.py``).
Uses networkx + pickle for DBLP citation/collaboration graphs.

Thread safety: Pickle data is cached at process level (_graph_cache) and shared
across threads. All query methods are read-only. check_edges() for AuthorNet
uses dict() copy before mutating, so the cached graph is never modified.
"""

import pickle
import threading
from pathlib import Path

# Process-level cache: (corpus_dir, graph_name) -> dict of 6 objects (shared, read-only)
_graph_cache: dict[tuple[str, str], dict] = {}
_graph_cache_lock = threading.Lock()


def _read_graph_from_disk(corpus_dir: Path, graph_name: str) -> dict:
    """Read graph data from disk and return as a dict of 6 objects."""
    if graph_name == "dblp":
        dblp_dir = corpus_dir / "dblp"

        with open(dblp_dir / "paper_net.pkl", "rb") as f:
            paper_net = pickle.load(f)  # noqa: S301

        with open(dblp_dir / "author_net.pkl", "rb") as f:
            author_net = pickle.load(f)  # noqa: S301

        with open(dblp_dir / "title2id_dict.pkl", "rb") as f:
            title2id_dict = pickle.load(f)  # noqa: S301

        with open(dblp_dir / "author2id_dict.pkl", "rb") as f:
            author2id_dict = pickle.load(f)  # noqa: S301

        with open(dblp_dir / "id2title_dict.pkl", "rb") as f:
            id2title_dict = pickle.load(f)  # noqa: S301

        with open(dblp_dir / "id2author_dict.pkl", "rb") as f:
            id2author_dict = pickle.load(f)  # noqa: S301

        return {
            "paper_net": paper_net,
            "author_net": author_net,
            "title2id_dict": title2id_dict,
            "author2id_dict": author2id_dict,
            "id2title_dict": id2title_dict,
            "id2author_dict": id2author_dict,
        }
    else:
        raise ValueError(f"Unknown graph: {graph_name}")


class GraphToolkit:
    """Manages DBLP graph state (PaperNet + AuthorNet)."""

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.paper_net = None
        self.author_net = None
        self.id2title_dict = None
        self.title2id_dict = None
        self.id2author_dict = None
        self.author2id_dict = None

    def load_graph(self, graph_name: str) -> str:
        """LoadGraph[dblp] — load DBLP paper and author networks.

        Uses process-level cache with double-checked locking so pickle files
        are read from disk at most once across all threads.
        """
        key = (str(self.corpus_dir), graph_name)
        if key not in _graph_cache:
            with _graph_cache_lock:
                if key not in _graph_cache:
                    _graph_cache[key] = _read_graph_from_disk(
                        self.corpus_dir, graph_name
                    )

        cached = _graph_cache[key]
        self.paper_net = cached["paper_net"]
        self.author_net = cached["author_net"]
        self.id2title_dict = cached["id2title_dict"]
        self.title2id_dict = cached["title2id_dict"]
        self.id2author_dict = cached["id2author_dict"]
        self.author2id_dict = cached["author2id_dict"]

        return "DBLP data is loaded, including two graphs: AuthorNet and PaperNet."

    def _resolve_graph(self, graph_name: str):
        """Return (graph, name_to_id, id_to_name) for given graph name."""
        if graph_name == "PaperNet":
            return self.paper_net, self.title2id_dict, self.id2title_dict
        elif graph_name == "AuthorNet":
            return self.author_net, self.author2id_dict, self.id2author_dict
        else:
            raise ValueError(f"Unknown graph name: {graph_name}")

    def check_neighbours(self, argument: str) -> str:
        """NeighbourCheck[GraphName, Node] — list node's neighbours."""
        graph_name, node = argument.split(", ", 1)
        graph, name2id, id2name = self._resolve_graph(graph_name)
        neighbours = [id2name[n] for n in graph.neighbors(name2id[node])]
        return str(neighbours)

    def check_nodes(self, argument: str) -> str:
        """NodeCheck[GraphName, Node] — return node attributes."""
        graph_name, node = argument.split(", ", 1)
        graph, name2id, _ = self._resolve_graph(graph_name)
        return str(graph.nodes[name2id[node]])

    def check_edges(self, argument: str) -> str:
        """EdgeCheck[GraphName, Node1, Node2] — return edge attributes."""
        graph_name, node1, node2 = argument.split(", ", 2)

        if graph_name == "PaperNet":
            graph = self.paper_net
            dictionary = self.title2id_dict
            edge = graph.edges[dictionary[node1], dictionary[node2]]
            return str(edge)
        elif graph_name == "AuthorNet":
            graph = self.author_net
            dictionary = self.author2id_dict
            edge = dict(graph.edges[dictionary[node1], dictionary[node2]])
            # Convert paper IDs to titles for readability
            if "papers" in edge:
                edge["papers"] = [
                    self.id2title_dict[pid] for pid in edge["papers"]
                ]
            return str(edge)
        else:
            raise ValueError(f"Unknown graph name: {graph_name}")

"""Dataset evaluators.

Importing this package registers all built-in evaluators (one per dataset)
into :mod:`sragents.evaluate.base`.
"""

from sragents.evaluate.datasets import (  # noqa: F401
    champ,
    logicbench,
    medcalcbench,
    theoremqa,
    toolqa,
)
from sragents.evaluate.datasets import bigcodebench  # noqa: F401

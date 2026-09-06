from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar


# === Recipe base class ===


class Recipe:
    """Base class for GRPO task recipes.

    Override :meth:`rewards` to return the list of reward functions and
    their weights. The other class attributes document what the recipe
    expects from the dataset; they are not auto-validated and
    ``system_prompt`` is not auto-injected.
    """

    #: Short human-readable description. Shown in logs when loaded.
    description: ClassVar[str] = ""

    #: Dataset columns the rewards expect. Documentation only.
    required_columns: ClassVar[tuple[str, ...]] = ()

    #: Recommended system prompt. Include it in your dataset prep.
    system_prompt: ClassVar[str | None] = None

    def rewards(self) -> list[tuple[Callable, float]]:
        """Return ``[(reward_fn, weight), ...]`` for this task.

        Subclasses must override. To extend a parent, spread
        ``super().rewards()`` and append new pairs.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.rewards() must be overridden and return a "
            "list of (callable, float) tuples."
        )

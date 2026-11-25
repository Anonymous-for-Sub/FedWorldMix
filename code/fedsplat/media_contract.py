"""Publication media layout validation shared by figure builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ALLOWED_PAPER_GRIDS = {
    "single-column": {(3, 2), (3, 3)},
    "double-column": {(7, 2), (8, 2)},
    "appendix": {(9, 2), (8, 3)},
}


@dataclass(frozen=True)
class PaperLayout:
    paper_width: str
    grid_columns: int
    grid_rows: int
    canonical_suffix: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_paper_layout(paper_width: str, grid_columns: int, grid_rows: int) -> PaperLayout:
    """Validate the mandatory columns-by-rows publication grid contract."""
    if paper_width not in ALLOWED_PAPER_GRIDS:
        raise ValueError(f"Unknown paper width {paper_width!r}; choose {sorted(ALLOWED_PAPER_GRIDS)}")
    grid = (int(grid_columns), int(grid_rows))
    if grid not in ALLOWED_PAPER_GRIDS[paper_width]:
        allowed = ", ".join(
            f"{columns}x{rows}" for columns, rows in sorted(ALLOWED_PAPER_GRIDS[paper_width])
        )
        raise ValueError(f"Grid {grid[0]}x{grid[1]} is not allowed for {paper_width}; choose {allowed}")
    prefix = {
        "single-column": "paper_singlecol",
        "double-column": "paper_doublecol",
        "appendix": "appendix",
    }[paper_width]
    return PaperLayout(
        paper_width=paper_width,
        grid_columns=grid[0],
        grid_rows=grid[1],
        canonical_suffix=f"__{prefix}_{grid[0]}x{grid[1]}",
    )

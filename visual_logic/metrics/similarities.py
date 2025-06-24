from typing import Optional

import numpy as np

from visual_logic.typing_and_extensions import Grid


def cells_simmilarity(
    grid0: Grid, grid1: Grid, ids_to_consider: Optional[list[int]] = None
):
    grid0 = np.array(grid0)
    grid1 = np.array(grid1)
    if grid0.shape != grid1.shape:
        return 0.0

    if ids_to_consider is not None:
        ids_to_consider = set(ids_to_consider)
        valid_mask = np.isin(grid0, list(ids_to_consider)) | np.isin(
            grid1, list(ids_to_consider)
        )
    else:
        valid_mask = np.ones_like(grid0, dtype=bool)

    return np.mean((grid0 == grid1)[valid_mask])


def discrete_similarity(grid0: Grid, grid1: Grid):
    grid0 = np.array(grid0)
    grid1 = np.array(grid1)
    return np.all(grid0 == grid1)

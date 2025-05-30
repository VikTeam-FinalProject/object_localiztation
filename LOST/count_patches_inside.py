import torch
import numpy as np
from typing import List, Tuple
def potentials_in_boxes(
    potentials: torch.Tensor,
    w_fmap: int,
    h_fmap: int,
    patch_size: int,
    gt_boxes: np.ndarray
) -> Tuple[List[int], int]:
    """
    Returns
    -------
    per_box_counts : List[int]
        per_box_counts[i] = number of potentials inside gt_boxes[i]
    outside_count : int
        number of potentials not in any gt box
    """
    # bring to CPU-long for masking
    potentials = torch.as_tensor(potentials, dtype=torch.long, device="cpu")

    # compute (x,y) of each patch-center
    rows = potentials // w_fmap
    cols = potentials %  w_fmap
    xc = (cols.float() + 0.5) * patch_size
    yc = (rows.float() + 0.5) * patch_size

    per_box_counts: List[int] = []
    covered = torch.zeros_like(potentials, dtype=torch.bool)

    for xmin, ymin, xmax, ymax in gt_boxes:
        mask = (xc >= xmin) & (xc <= xmax) & (yc >= ymin) & (yc <= ymax)
        per_box_counts.append(int(mask.sum()))
        covered |= mask

    outside_count = int((~covered).sum())
    return per_box_counts, outside_count
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

def get_patches_in_mask(mask: np.ndarray,
                        patch_size: int,
                        w_fmap: int,
                        h_fmap: int) -> np.ndarray:
    """
    mask: H x W binary (1=foreground) numpy array
    patch_size: size of one square patch in pixels
    w_fmap, h_fmap: number of patches along height and width
    returns: 1D array of length w_fmap*h_fmap, True if that patch-center is in mask
    """
    # build grid of patch-centers
    # row idx goes 0..w_fmap-1, col idx 0..h_fmap-1
    rows = np.repeat(np.arange(w_fmap), h_fmap)
    cols = np.tile(np.arange(h_fmap), w_fmap)
    # center coords
    ys = (rows + 0.5) * patch_size
    xs = (cols + 0.5) * patch_size
    # clamp in case image not exactly multiple
    ys = np.minimum(ys.astype(int), mask.shape[0]-1)
    xs = np.minimum(xs.astype(int), mask.shape[1]-1)
    inside = mask[ys, xs] > 0
    return inside  # boolean array length w_fmap*h_fmap

def count_semantic_patches(potentials: List[int],
                           mask: np.ndarray,
                           patch_size: int,
                           w_fmap: int,
                           h_fmap: int) -> Tuple[int,int]:
    """
    potentials: list of top-k patch indices (0 .. w_fmap*h_fmap-1)
    mask: H x W binary semantic mask
    returns (total_in_semantics, potentials_in_semantics)
    """
    inside_all = get_patches_in_mask(mask, patch_size, w_fmap, h_fmap)
    total_in = int(inside_all.sum())
    # now see which of your potentials land in the mask
    pot_array = np.array(potentials, dtype=int)
    total_pot_in = int(inside_all[pot_array].sum())
    return total_in, total_pot_in
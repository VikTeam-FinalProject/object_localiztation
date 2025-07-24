# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import scipy.ndimage
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


def visualize_index(N,idxs):
    flat = idxs.reshape(-1).cpu()

    counts = torch.bincount(flat, minlength=N)  # [N]

    # Vẽ bar chart
    x = torch.arange(N).numpy()
    y = counts.numpy()

    plt.figure(figsize=(6, 4))
    plt.bar(x, y, color='c')
    plt.xlabel("Index")
    plt.ylabel("Số cặp có chứa index")
    plt.title("Tần suất xuất hiện của mỗi index")
    plt.xticks(x)
    plt.grid(axis='y', alpha=0.5)
    plt.savefig("counts.png")

def test_magnitude_vs_dot(A,feats, feats_):
    """
    feats, feats_: [1, N, D]  (batch size = 1)
    Trả về:
      thresh: ngưỡng mean(dot)
      idxs:   tensor [K,2] các cặp (i,j) sao cho dot>thresh
      dot:    tensor [K]           các giá trị dot tương ứng
      norm:   tensor [K]           các giá trị ||v_i||*||v_j||
      cos:    tensor [K]           các giá trị cosine = dot/norm
    """
    # 1) Dot-matrix [N,N]
    # 2) Norm-vectors [N]
    N = feats.shape[1]  # [1, N, D]
    norms_f  = feats.norm(p=2, dim=-1).squeeze(0)     # [N]
    norms_f_ = feats_.norm(p=2, dim=-1).squeeze(0)    # [N]

    # 3) Norm-product matrix [N,N]
    P = norms_f.unsqueeze(1) * norms_f_.unsqueeze(0)  # [N, N]
    print(P.shape)
    # 4) Cosine matrix [N,N]
    C = A / (P + 1e-8)

    # 5) Threshold
    thresh = A.mean()
    # 6) Mask và advanced indexing
    mask = torch.triu(P, diagonal=1) > thresh
    idxs = mask.nonzero(as_tuple=False)
    num_selected = mask.sum().item()
    torch.set_printoptions(profile="full")
    # 2) Tổng số cặp
    total_pairs = N*(N-1)/2
    # 3) Tỷ lệ %
    ratio = num_selected / total_pairs * 100.0
    visualize_index(N,idxs)

    # exit()
    dot_vals      = A[mask]         # [K]
    norm_vals     = P[mask]         # [K]
    cosine_vals   = C[mask]         # [K]

    return thresh, idxs, dot_vals, norm_vals, cosine_vals

def check_affinity_outliers(A: torch.Tensor, threshold_std: float = 3.0) -> torch.Tensor:
    # Flatten A and detect outliers (ignore diagonal if needed)
    A_flat = A.flatten()

    mean, std = A_flat.mean(), A_flat.std()
    outliers = (A_flat - mean).abs() > threshold_std * std
    return outliers.reshape(A.shape)

# def lost(feats,feats_, dims, scales, init_image_size, k_patches=100, dynamic_thres=False, artifact_idx=None):
#     """
#     Implementation of LOST method.
#     Inputs
#         feats: the pixel/patch features of an image
#         dims: dimension of the map from which the features are used
#         scales: from image to map scale
#         init_image_size: size of the image
#         k_patches: number of k patches retrieved that are compared to the seed at seed expansion
#     Outputs
#         pred: box predictions
#         A: binary affinity matrix
#         scores: lowest degree scores for all patches
#         seed: selected patch corresponding to an object
#     """
#     A = (feats @ feats_.transpose(1, 2)).squeeze()
#     affinity_outliers = check_affinity_outliers(A)

#     diag = A.diag()
#     off_diag = A - torch.diag(diag)
#     #thresh, idxs, dot_vals, norm_vals, cosine_vals = test_magnitude_vs_dot(A, feats, feats_)

#     #test_magnitude_vs_dot(A, feats, feats_)
#     sorted_patches, scores, jumps = patch_scoring(A, dynamic_thres, k_patches=k_patches, ar_idx=artifact_idx)

#     seed = sorted_patches[0] if len(sorted_patches) > 0 else 0
#     if k_patches == -1:
#         not_potentials_xy = [np.unravel_index(p.cpu(), dims) for p in sorted_patches]
#         not_potentials_filtered_index = [np.ravel_multi_index(p, dims) for p in not_potentials_xy]
#     else:
#         potentials = sorted_patches[:k_patches]
#         # Should be empty or very small
#         not_potentials_xy = [np.unravel_index(p.cpu(), dims) for p in potentials]
#         not_potentials_filtered_index = [np.ravel_multi_index(p, dims) for p in not_potentials_xy]
#     pred, _ = detect_box(dims, scales=scales, object_patches=not_potentials_filtered_index,
#                          initial_im_size=init_image_size[1:])
    
#     return np.asarray(pred), A, scores, seed, not_potentials_filtered_index, jumps #, idxs

def lost(
    feats, feats_, dims, scales, init_image_size,
    k_patches=100, dynamic_thres=False, artifact_idx=None,
    custom_scores = None,  # NEW arg
):
    """
    Implementation of LOST method with optional custom_scores override.
    """
    # 1️⃣ If the user passed us fused attention scores, skip affinity/patch_scoring
    if custom_scores is not None:
        # custom_scores: shape (N_patches,) on torch device
        scores = custom_scores.detach().numpy()  # to numpy for downstream
        # argsort ascending → lowest‐degree first
        sorted_patches = list(np.argsort(scores))
        jumps = None
    else:
        # ⚙️ original LOST path
        A = (feats @ feats_.transpose(1, 2)).squeeze()
        affinity_outliers = check_affinity_outliers(A)

        # drop self‐similarity
        diag = A.diag()
        off_diag = A - torch.diag(diag)

        sorted_patches, scores, jumps = patch_scoring(
            A, dynamic_thres, k_patches=k_patches, ar_idx=artifact_idx
        )
        scores = scores.cpu().numpy()
        sorted_patches = [int(p) for p in sorted_patches]

    seed = sorted_patches[0] if sorted_patches else 0

    # 2️⃣ derive the “potential” patch indices exactly as before
    if k_patches == -1:
        chosen = sorted_patches
    else:
        chosen = sorted_patches[:k_patches]

    # convert flat indices → box patches
    not_potentials_xy = [np.unravel_index(p, dims) for p in chosen]
    not_potentials_filtered_index = [
        np.ravel_multi_index(xy, dims) for xy in not_potentials_xy
    ]

    # 3️⃣ final box from LOST’s detect_box
    pred, _ = detect_box(
        dims,
        scales=scales,
        object_patches=not_potentials_filtered_index,
        initial_im_size=init_image_size[1:],
    )

    return (
        np.asarray(pred),
        None if custom_scores is not None else A,  # affinity only if original path
        scores,
        seed,
        not_potentials_filtered_index,
        jumps,
    )

def compute_dynamic_k(image_size, base_k=100, scale_factor=0.0005):
    """
    Compute k_patches based on image area.
    Args:
        image_size: Tuple (height, width) of the image.
        base_k: Base number of patches (for a reference image size).
        scale_factor: Scaling factor to adjust k_patches proportionally.
    Returns:
        Adjusted k_patches (int).
    """
    height, width = image_size
    image_area = height * width
    dynamic_k = int(scale_factor * image_area)
    return max(dynamic_k, 10)

def compute_k_from_bboxes(gt_bbxs, image_size, base_k=25):
    """
    Compute dynamic k_patches based on GT bounding boxes.
    
    Args:
        gt_bbxs: np.array of shape (N, 4) - bounding boxes
        image_size: tuple (H, W)
        base_k: starting value to scale from

    Returns:
        int - dynamically computed k_patches
    """
    if gt_bbxs is None or len(gt_bbxs) == 0:
        return base_k

    H, W = image_size
    img_area = H * W

    # Calculate area for each box
    box_areas = (gt_bbxs[:, 2] - gt_bbxs[:, 0]) * (gt_bbxs[:, 3] - gt_bbxs[:, 1])
    total_bbox_area = np.sum(box_areas)
    box_density = len(gt_bbxs) / (H * W)

    # Adjust k using both density and area coverage
    area_ratio = total_bbox_area / img_area
    scale_factor = (box_density * 1e5) + area_ratio

    dynamic_k = int(base_k * (1 + scale_factor * 2.0))
    return max(25, min(dynamic_k, 500))

def dbscan_filter(patches_xy):
    if len(patches_xy) == 0:
        return []
    clustering = DBSCAN(eps=1, min_samples=5).fit(patches_xy)
    labels = clustering.labels_
    from collections import Counter
    element_counts = Counter(labels)
    most_common_element, count = element_counts.most_common(1)[0]
    if most_common_element == -1:
        try:
            most_common_element, _ = element_counts.most_common(2)[1]
        except:
            return []
    return [patches_xy[i] for i in range(len(labels)) if labels[i] == most_common_element]


def patch_scoring(M, dynamic_threshold, k_patches, ar_idx=None):
    """
    Patch scoring based on the inverse degree.
        dynamic_threshold: set to True will override the threshold value by mean of the matrix
    """
    threshold = torch.mean(M) if dynamic_threshold else 0.0
    A = M.clone()
    A.fill_diagonal_(0)
    cent = torch.sum(A > threshold, dim=1).float()
    total_elements = A.numel() - A.shape[0]  # trừ đi phần tử đường chéo
    num_above_threshold = torch.sum(A > threshold).item()
    # Tính tỉ lệ phần trăm
    percent_above = (num_above_threshold / total_elements) * 100
    sel = torch.argsort(cent, descending=False)
    cent = cent[sel]
    jumps = []
    if k_patches == -1:
        jumps = [abs(int(float((cent[i] - cent[i - 1]).cpu().numpy()))) for i in range(1, len(cent))]
        # plot_jumps(jumps)
        # replace first 10% of the jumps with 0
        num_10_percent = int(len(jumps) * 0.1)
        jumps[:num_10_percent] = [0]*num_10_percent

        # replace last 10% of the jumps with 0
        jumps[-num_10_percent:] = [0]*num_10_percent
        
        k_jump = get_last_argmax(jumps)       
        # if len(jumps) > 10:
        #     k_jump = 10 + np.argmin(jumps[10:-10])
        # else:
        #     k_jump = np.argmin(jumps)
        return sel[:k_jump], cent, jumps

    return sel, cent, jumps

def get_last_argmax(lst):
    '''
        [1, 4, 4, 4, 3, 4] --> return 5
        default argmax would return 1
    '''
    lst2 = lst[::-1]
    return len(lst) - lst2.index(max(lst)) -1

def plot_jumps(jumps,save_path='jumps.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(jumps, marker='o', linestyle='-', color='b')
    plt.title('Plot of Jumps')
    plt.xlabel('Index')
    plt.ylabel('Jump Value')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

def detect_box(dims, object_patches, initial_im_size=None, scales=None):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    if isinstance(object_patches, torch.Tensor):
        if object_patches.dim() == 0:  # It's a scalar
            return [0, 0, 0, 0], [0, 0, 0, 0]
        object_patches = object_patches.tolist()  # Convert to list

    if len(object_patches) == 0:
        return [0, 0, 0, 0], [0, 0, 0, 0]
    object_patches_unravel = [np.unravel_index(p, dims) for p in object_patches]
    mask = np.zeros(dims)
    for patch_id in object_patches_unravel:
        mask[patch_id] = 1

    mask = np.where(mask == 1)
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])

    return pred, [ymin, xmin, ymax, xmax]


def dino_seg(attn, dims, patch_size, head=0):
    """
    Extraction of boxes based on the DINO segmentation method proposed in https://github.com/facebookresearch/dino.
    Modified from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
    """
    w_featmap, h_featmap = dims
    nh = attn.shape[1]
    official_th = 0.6

    attentions = attn[0, :, 0, 1:].reshape(nh, -1)
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - official_th)
    idx2 = torch.argsort(idx)
    for h in range(nh):
        th_attn[h] = th_attn[h][idx2[h]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

    labeled_array, num_features = scipy.ndimage.label(th_attn[head].cpu().numpy())
    size_components = [np.sum(labeled_array == c) for c in range(np.max(labeled_array))]

    biggest_component = np.argmax(size_components[1:]) + 1 if len(size_components) > 1 else 0
    mask = np.where(labeled_array == biggest_component)

    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    r_xmin, r_xmax = xmin * patch_size, xmax * patch_size
    r_ymin, r_ymax = ymin * patch_size, ymax * patch_size
    return [r_xmin, r_ymin, r_xmax, r_ymax]


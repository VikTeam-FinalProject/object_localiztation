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
from datasets import bbox_iou


def lost(feats, dims, scales, init_image_size, k_patches=100, dynamic_thres=False, dbscan=True):
    """
    Implementation of LOST method.
    Inputs
        feats: the pixel/patch features of an image
        dims: dimension of the map from which the features are used
        scales: from image to map scale
        init_image_size: size of the image
        k_patches: number of k patches retrieved that are compared to the seed at seed expansion
    Outputs
        pred: box predictions
        A: binary affinity matrix
        scores: lowest degree scores for all patches
        seed: selected patch corresponding to an object
    """
    A = (feats @ feats.transpose(1, 2)).squeeze()
    sorted_patches, scores = patch_scoring(A, dynamic_thres, k_patches=k_patches)
    seed = sorted_patches[-1] if len(sorted_patches) > 0 else 0

    if k_patches == -1:
        not_potentials_xy = [np.unravel_index(p.cpu(), dims) for p in sorted_patches]
        not_potentials_xy_filtered = dbscan_filter(not_potentials_xy) if dbscan else not_potentials_xy
        not_potentials_filtered_index = [np.ravel_multi_index(p, dims) for p in not_potentials_xy_filtered]
    else:
        potentials = sorted_patches[:k_patches]
        not_potentials_xy = [np.unravel_index(p.cpu(), dims) for p in potentials]
        not_potentials_xy_filtered = dbscan_filter(not_potentials_xy) if dbscan else not_potentials_xy
        not_potentials_filtered_index = [np.ravel_multi_index(p, dims) for p in not_potentials_xy_filtered]

    pred, _ = detect_box(dims, scales=scales, object_patches=not_potentials_filtered_index,
                         initial_im_size=init_image_size[1:])

    return np.asarray(pred), A, scores, seed, not_potentials_filtered_index


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


def patch_scoring(M, dynamic_threshold, k_patches):
    """
    Patch scoring based on the inverse degree.
        dynamic_threshold: set to True will override the threshold value by mean of the matrix
    """
    threshold = torch.mean(M) if dynamic_threshold else 0.0
    A = M.clone()
    A.fill_diagonal_(0)
    A[A < threshold] = 0
    cent = -torch.sum(A > threshold, dim=1).float()
    sel = torch.argsort(cent, descending=True)
    cent = cent[sel]

    # Dynamic Patch Selection (if `k_patches == -1`)
    if k_patches == -1:
        jumps = [int(float((cent[i] - cent[i - 1]).cpu().numpy())) for i in range(1, len(cent))]
        plot_jumps(jumps)

        if len(jumps) > 10:
            k_jump = 10 + np.argmin(jumps[10:])
        else:
            k_jump = np.argmin(jumps)
        return sel[:k_jump], cent

    return sel, cent


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

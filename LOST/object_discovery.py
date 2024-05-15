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
import scipy
import scipy.ndimage
from sklearn.cluster import DBSCAN
import numpy as np
from datasets import bbox_iou


def lost(feats, dims, scales, init_image_size, k_patches=100, dynamic_thres=False, dbscan = True):
    """
    Implementation of LOST method.
    Inputs
        feats: the pixel/patche features of an image
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
    # Compute the similarity
    A = (feats @ feats.transpose(1, 2)).squeeze()
    # Compute the inverse degree centrality measure per patch
    sorted_patches, scores = patch_scoring(A, dynamic_thres)
    num_0_score = len([s for s in scores if s == 0])
    # Select the initial seed
    seed = sorted_patches[-1]

    # Seed expansion
    potentials = sorted_patches[:k_patches]

    not_potentials = [p for p in sorted_patches if p not in potentials]
    not_potentials_xy = [np.unravel_index(p.cpu(), (dims[0], dims[1])) for p in not_potentials]
    print('Number of not_potentials patches:', len(not_potentials))
    if dbscan:
        print('DBSCAN clustering')
        # dbscan: clustering of the object patches, keep the largest cluster
        if len(not_potentials_xy) == 0:
            # no object-patch
            labels = []
        else:
            # len > 0
            clustering = DBSCAN(eps=1, min_samples=5).fit(not_potentials_xy)
            labels = clustering.labels_
            # get id of the largest cluster
            from collections import Counter
            element_counts = Counter(labels)
            print("dbscan element counts:", element_counts)
            most_common_element, count = element_counts.most_common(1)[0]
            if most_common_element == -1:
                try:
                    most_common_element, _ = element_counts.most_common(2)[1]
                except:
                    # no object, all are noise
                    labels = []
        not_potentials_xy_filtered = [not_potentials_xy[i] for i in range(len(labels)) if labels[i] == most_common_element]
    else:
        # turn off dbscan
        not_potentials_xy_filtered = not_potentials_xy
        
    not_potentials_filtered_index = [np.ravel_multi_index((p[0], p[1]), (dims[0], dims[1])) for p in not_potentials_xy_filtered]
    
    similars = potentials[A[seed, potentials] > 0.0]
    M = torch.sum(A[not_potentials_filtered_index, :], dim=0)
    print('len not_potentials_filtered_index:', len(not_potentials_filtered_index))
    # Box extraction
    pred, _ = detect_box(
        M, seed, dims, scales=scales, object_patches = not_potentials_filtered_index, initial_im_size=init_image_size[1:]
    )

    return np.asarray(pred), A, scores, seed, not_potentials_filtered_index, similars


def patch_scoring(M, dynamic_threshold: bool):
    """
    Patch scoring based on the inverse degree.
        dynamic_threshold: set to True will override the threshold value by mean of the matrix
    """
    if dynamic_threshold:
        threshold = torch.mean(M)
    else:
        threshold = 0.0
    # Cloning important
    A = M.clone()

    # Zero diagonal
    A.fill_diagonal_(0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.t()

    # Sort pixels by inverse degree
    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    sel = torch.argsort(cent, descending=True)

    return sel, cent


def detect_box(A, seed, dims, object_patches, initial_im_size=None, scales=None):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    if len(object_patches) == 0:
        # no object
        return [0, 0, 0, 0], [0,0,0,0]
    object_patches_unravel = [np.unravel_index(p, dims) for p in object_patches]
    mask = np.zeros(dims)
    for patch_id in object_patches_unravel:
        mask[patch_id] = 1
    # w_featmap, h_featmap = dims

    # correl = A.reshape(w_featmap, h_featmap).float()
    # Compute connected components
    # labeled_array, num_features = scipy.ndimage.label(correl.cpu().numpy() > 0.0)
    # Find connected component corresponding to the initial seed
    # cc = labeled_array[np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))]

    # Should not happen with LOST
    # if cc == 0:
    #     pass
        # raise ValueError("The seed is in the background component.")

    # Find box
    mask = np.where(mask == 1)
    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    # Rescale to image size
    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax

    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])

    # Coordinate predictions for the feature space
    # Axis different then in image space
    pred_feats = [ymin, xmin, ymax, xmax]

    return pred, pred_feats


def dino_seg(attn, dims, patch_size, head=0):
    """
    Extraction of boxes based on the DINO segmentation method proposed in https://github.com/facebookresearch/dino. 
    Modified from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
    """
    w_featmap, h_featmap = dims
    nh = attn.shape[1]
    official_th = 0.6

    # We keep only the output patch attention
    # Get the attentions corresponding to [CLS] token
    attentions = attn[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - official_th)
    idx2 = torch.argsort(idx)
    for h in range(nh):
        th_attn[h] = th_attn[h][idx2[h]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

    # Connected components
    labeled_array, num_features = scipy.ndimage.label(th_attn[head].cpu().numpy())

    # Find the biggest component
    size_components = [np.sum(labeled_array == c) for c in range(np.max(labeled_array))]

    if len(size_components) > 1:
        # Select the biggest component avoiding component 0 corresponding to background
        biggest_component = np.argmax(size_components[1:]) + 1
    else:
        # Cases of a single component
        biggest_component = 0

    # Mask corresponding to connected component
    mask = np.where(labeled_array == biggest_component)

    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    # Rescale to image
    r_xmin, r_xmax = xmin * patch_size, xmax * patch_size
    r_ymin, r_ymax = ymin * patch_size, ymax * patch_size
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    return pred
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

import numpy as np
from datasets import bbox_iou


def lost(feats, dims, scales, init_image_size, k_patches=100):
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
    print("feats.shape", feats.shape)
    feats_binary = feats.clone()
    feats_binary[feats_binary>0] = 0
    feats_binary[feats_binary<0] = 1
    feats_binary = feats_binary.cpu().numpy()*255
    feats_binary = np.expand_dims(feats_binary.squeeze(), -1)
    print('feats bin', feats_binary)
    print('feats_binary.shape', feats_binary.shape)
    import cv2
    cv2.imwrite(f'feats_binary_{feats.shape[1]}.png', feats_binary)
    
    feat_file = 'feat_tensor.txt'
    with open(feat_file, 'w') as f:
        for line in feats[0]:
            f.write(str(line) + '\n')
    print('init_image_size', init_image_size)
    print('dims', dims)
    # Compute the similarity
    print('feats is: ', feats) # 1, 672, 384
    print('sum feature of first patch: ', torch.sum(feats[0][0]))   # 1 con so
    A = (feats @ feats.transpose(1, 2)).squeeze()
    print('A is: ', A)
    # Compute the inverse degree centrality measure per patch
    sorted_patches, scores = patch_scoring(A)
    print('shap of sorted_patches', sorted_patches.shape)
    print('sorted_patches', sorted_patches)
    # Select the initial seed
    seed = sorted_patches[0]

    # Seed expansion
    potentials = sorted_patches[:k_patches]
    similars = potentials[A[seed, potentials] > 0.0]
    M = torch.sum(A[similars, :], dim=0)

    # Box extraction
    pred, _ = detect_box(
        M, seed, dims, scales=scales, initial_im_size=init_image_size[1:]
    )

    return np.asarray(pred), A, scores, seed


def patch_scoring(M, threshold=0.):
    """
    Patch scoring based on the inverse degree.
    """
    # Cloning important
    A = M.clone()
    # Zero diagonal
    A.fill_diagonal_(0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.t()

    # Sort pixels by inverse degree
    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    print('cent', cent)
    sel = torch.argsort(cent, descending=True)

    return sel, cent


def detect_box(A, seed, dims, initial_im_size=None, scales=None):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims # (24, 32)
    print('A 6 6: ', A[ :6])
    correl = A.reshape(w_featmap, h_featmap).float()
    print('correl.shape', correl.shape,  '\n',correl[:6, :6])
    # Compute connected components
    labeled_array, num_features = scipy.ndimage.label(correl.cpu().numpy() > 0.0)
    print('num_features', num_features)
    print('labeled_array.shape', labeled_array.shape, '\n', labeled_array[:6, :6])
    # Find connected component corresponding to the initial seed
    print('seed shape', seed.shape)
    print('seed', seed)
    seed_xy = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
    print('tmp_xyz', seed_xy)
    seed_type_fore_or_background = labeled_array[seed_xy]
    print('cc', seed_type_fore_or_background)
    
    # Should not happen with LOST
    if seed_type_fore_or_background == 0:
        raise ValueError("The seed is in the background component.")

    # Find box
    mask = np.where(labeled_array == seed_type_fore_or_background)
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

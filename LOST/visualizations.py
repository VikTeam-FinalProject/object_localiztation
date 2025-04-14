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

import cv2
import torch
import skimage.io
import numpy as np
import torch.nn as nn
from PIL import Image
import os
import matplotlib.pyplot as plt

def visualize_predictions(image, pred, seed, scales, dims, vis_folder, im_name, plot_seed=False, potentials=None):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    w_featmap, h_featmap = dims
    # Plot the box
    im_name = im_name.split("\\")[-1]
    if plot_seed:
        if type(seed) == torch.Tensor:
            s_ = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
        else:
            s_ = np.unravel_index(seed, (w_featmap, h_featmap))
        size_ = np.asarray(scales) / 2
        cv2.rectangle(
            image,
            (int(s_[1] * scales[1] - (size_[1] / 2)), int(s_[0] * scales[0] - (size_[0] / 2))),
            (int(s_[1] * scales[1] + (size_[1] / 2)), int(s_[0] * scales[0] + (size_[0] / 2))),
            (0, 255, 0), -1,
        )



    if potentials is not None:
        # plot half of potentials
        for seed in potentials[:len(potentials)//2]:
            if type(seed) == torch.Tensor:
                s = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
            else:
                s = np.unravel_index(seed, (w_featmap, h_featmap))
            size = np.asarray(scales) / 2
            cv2.rectangle(
                image,
                (int(s[1] * scales[1] - (size[1] / 2)), int(s[0] * scales[0] - (size[0] / 2)),
                ),
                (int(s[1] * scales[1] + (size[1] / 2)), int(s[0] * scales[0] + (size[0] / 2)),
                ),
                (128, 0, 128), 1, # Purple
            )
        # plot the other half of potentials
        for seed in potentials[len(potentials)//2:]:
            if type(seed) == torch.Tensor:
                s = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
            else:
                s = np.unravel_index(seed, (w_featmap, h_featmap))
            size = np.asarray(scales) / 2
            cv2.rectangle(
                image,
                (int(s[1] * scales[1] - (size[1] / 2)), int(s[0] * scales[0] - (size[0] / 2)),
                ),
                (int(s[1] * scales[1] + (size[1] / 2)), int(s[0] * scales[0] + (size[0] / 2)),
                ),
                (0, 128, 128), 1, # Teal
            )
        
        # visualize paths at id= k_jump
        patch_k_jump = potentials[-1] if len(potentials) > 0 else 0
        s = np.unravel_index(patch_k_jump, (w_featmap, h_featmap))
        size = np.asarray(scales) / 2

        pltname = f"{vis_folder}/LOST_{im_name}_potentials.png"
        Image.fromarray(image).save(pltname)
        # print(f"Predictions saved at {pltname}.")
        #
        # os.makedirs(f"{vis_folder}/jumps", exist_ok=True)
        # # save jumps plot
        # plt.figure(figsize=(10, 6))
        # plt.plot(jumps, marker='o', linestyle='-', color='b')
        # plt.title('Plot of Jumps')
        # plt.xlabel('Index')
        # plt.ylabel('Jump Value')
        # plt.grid(True)
        # plt.savefig(f"{vis_folder}/jumps/jumps_{im_name}.png")
        

def visualize_fms(A, seed, scores, dims, scales, output_folder, im_name):
    """
    Visualization of the maps presented in Figure 2 of the paper. 
    """
    w_featmap, h_featmap = dims

    # Binarized similarity
    binA = A.copy()
    binA[binA < 0] = 0
    binA[binA > 0] = 1

    # Get binarized correlation for this pixel and make it appear in gray
    im_corr = np.zeros((3, len(scores)))
    where = binA[seed, :] > 0
    im_corr[:, where] = np.array([128 / 255, 133 / 255, 133 / 255]).reshape((3, 1))
    # Show selected pixel in green
    im_corr[:, seed] = [204 / 255, 37 / 255, 41 / 255]
    # Reshape and rescale
    im_corr = im_corr.reshape((3, w_featmap, h_featmap))
    im_corr = (
        nn.functional.interpolate(
            torch.from_numpy(im_corr).unsqueeze(0),
            scale_factor=scales,
            mode="nearest",
        )[0].cpu().numpy()
    )

    # Save correlations
    im_corr = (im_corr * 255).astype(np.uint8)  # Chuyển về uint8
    skimage.io.imsave(
        fname=f"{output_folder}/corr_{im_name}.png",
        arr=im_corr.transpose((1, 2, 0)),
    )
    print(f"Image saved at {output_folder}/corr_{im_name}.png .")

    # Save inverse degree
    im_deg = (
        nn.functional.interpolate(
            torch.from_numpy(1 / binA.sum(-1)).reshape(1, 1, w_featmap, h_featmap),
            scale_factor=scales,
            mode="nearest",
        )[0][0].cpu().numpy()
    )
    plt.imsave(fname=f"{output_folder}/deg_{im_name}.png", arr=im_deg)
    print(f"Image saved at {output_folder}/deg_{im_name}.png .")

def visualize_seed_expansion(image, pred, seed, pred_seed, scales, dims, vis_folder, im_name):
    """
    Visualization of the seed expansion presented in Figure 3 of the paper. 
    """
    w_featmap, h_featmap = dims

    # Before expansion
    cv2.rectangle(
        image,
        (int(pred_seed[0]), int(pred_seed[1])),
        (int(pred_seed[2]), int(pred_seed[3])),
        (204, 204, 0),  # Yellow
        3,
    )

    # After expansion
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (204, 0, 204),  # Magenta
        3,
    )

    # Position of the seed
    center = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
    start_1 = center[0] * scales[0]
    end_1 = center[0] * scales[0] + scales[0]
    start_2 = center[1] * scales[1]
    end_2 = center[1] * scales[1] + scales[1]
    image[start_1:end_1, start_2:end_2, 0] = 204
    image[start_1:end_1, start_2:end_2, 1] = 37
    image[start_1:end_1, start_2:end_2, 2] = 41

    pltname = f"{vis_folder}/LOST_seed_expansion_{im_name}.png"
    Image.fromarray(image).save(pltname)
    print(f"Image saved at {pltname}.")


def visualize_heatmap(heatmap, im_name, vis_folder, mean=False):
    """
    Visualization of the heatmap.

    Args:
        heatmap: The heatmap to visualize.
        im_name: The name of the image.
        vis_folder: The folder to save the visualization.
        mean: Whether to visualize the mean of the heatmap.
    """
    save_dir = os.path.join(vis_folder, "heatmap")
    os.makedirs(save_dir, exist_ok=True)

    def normalize(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    if mean:
        if heatmap.ndim == 3:
            heatmap_mean = heatmap.mean(axis=0)
        else:
            heatmap_mean = heatmap
        heatmap_mean = normalize(heatmap_mean)
        fname = os.path.join(save_dir, f"attn-head_mean_{im_name}.png")
        plt.imsave(fname=fname, arr=heatmap_mean, format='png')
        print(f"{fname} saved.")
    else:
        for j in range(heatmap.shape[0]):
            head_map = normalize(heatmap[j])
            fname = os.path.join(save_dir, f"attn-head_{j}_{im_name}.png")
            plt.imsave(fname=fname, arr=head_map, format='png')
            print(f"{fname} saved.")



# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note the original is here: https://github.com/facebookresearch/dino/blob/main/visualize_attention.py

import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

from pathlib import Path
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
parent_dir = Path("model").resolve().parent.parent

sys.path.append(str(parent_dir))
strr = '\n'.join(sys.path)
sys.path.remove('/home/thekhoi/anaconda3/envs/env_dinoconda/lib/python3.9/site-packages/dinov2-0.0.1-py3.9.egg')
print(strr)
from dinov2.models.vision_transformer import vit_small, vit_large


if __name__ == '__main__':
    image_size = (952, 952)
    output_dir = 'output'
    patch_size = 14
    num_register_tokens = 4
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("init model")
    model = vit_small(
            patch_size=14,
            img_size=526,
            init_values=1.0,
            num_register_tokens=num_register_tokens,
            #ffn_layer="mlp",
            block_chunks=0
    )
    HOME = os.getcwd()

    MODEL_PATH = os.path.join(parent_dir,"model", 'dinov2_vits14_reg4_pretrain.pth')
    print('model path is: ', MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda:0'))
    for p in model.parameters():
        p.requires_grad = False

    model.to(device)
    model.eval()

    img = Image.open('/home/thekhoi/futme/RTD-ueh/object_localiztation/LOST/examples/VOC07_000236.jpg')
    img = img.convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    print(img.shape)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    print(img.shape)
    # print('model ', model)
    # print(dir(model))
    attentions = model.get_last_self_attention(img.to(device))
    print('attn shape ', attentions.shape)
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    # for every patch
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    # weird: one pixel gets high attention over all heads?
    print(torch.max(attentions, dim=1))
    attentions[:, 283] = 0
    print('attn shape: ', attentions.shape)
    attentions = attentions[:, :-num_register_tokens].reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)

    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

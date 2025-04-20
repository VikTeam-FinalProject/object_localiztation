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
import os

import torch
import torch.nn as nn
import sys
import dino.vision_transformer as vits

from dinov2.models.vision_transformer import vit_small,vit_large
def get_model(arch, patch_size, device):
    if "dinov2_vits14_pretrain" in arch:
        model = vit_small(patch_size=14,       
                    img_size=526,
                    init_values=1.0,
                    block_chunks=0
                )
    elif "dinov2_vitl14_pretrain" in arch:
        model = vit_large(patch_size=14,
                    img_size=526,
                    init_values=1.0,
                    block_chunks=0
                )
    elif "dinov2_vitl14_reg4_pretrain" in arch:
        model = vit_large(patch_size=14,
                    img_size=526,
                    init_values=1.0,
                    block_chunks=0,
                    num_register_tokens=4
                )
    else:
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

    for p in model.parameters():
        p.requires_grad = False

    # Initialize model with pretraining
    if "imagenet" not in arch:
        url = None
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif "dinov2" in arch:
            print("loading dinov2...")
            HOME = os.getcwd()
            MODEL_PATH = os.path.join(HOME,"dinov2_model", f"{arch}.pth")
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda:0'))
            for p in model.parameters():
                p.requires_grad = False

            model.eval()
            model.to(device)
            return model
        if url is not None:
            print(
                "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            msg = model.load_state_dict(state_dict, strict=True)
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    url, msg
                )
            )
        else:
            print(
                "There is no reference weights available for this model => We use random weights."
            )

    # If ResNet or VGG16 loose the last fully connected layer

    model.eval()
    model.to(device)
    return model

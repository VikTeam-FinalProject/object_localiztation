# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)
from PIL import Image, UnidentifiedImageError
import numpy as np

def safe_image_open(image):
    from PIL import Image, UnidentifiedImageError
    import numpy as np

    # If it's already a PIL image, just return it safely
    if isinstance(image, Image.Image):
        try:
            return image.convert("RGB")
        except Exception:
            return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    # Otherwise, open from path
    try:
        with Image.open(image) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        # Return black placeholder if corrupt
        return None  # signal skip
    return None

logger = logging.getLogger("dinov2")
class SafeColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    def get_params(self, brightness, contrast, saturation, hue):
        # Clamp hue factor safely to [-0.5, 0.5]
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            super().get_params(brightness, contrast, saturation, hue)
        if isinstance(hue_factor, tuple):
            hue_factor = tuple(max(-0.5, min(0.5, f)) for f in hue_factor)
        else:
            hue_factor = max(-0.5, min(0.5, hue_factor))
        return fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)],
                #[SafeColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],    
                p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        image = safe_image_open(image)
        if image is None:
           raise StopIteration 
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

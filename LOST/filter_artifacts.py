import argparse
import os

import numpy as np
import torch

from datasets import ImageDataset, Dataset
from networks import get_model
from artifact_det import detect_artifacts



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ArtifactChecker",
        description="Check artifacts in model outputs and optionally visualize heatmaps."
    )

    parser.add_argument(
        "--arch",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "resnet50",
            "vgg16_imagenet",
            "resnet50_imagenet",
            "dinov2_vitl14_pretrain",
            "dinov2_vits14_pretrain",
            "dinov2_vitl14_reg4_pretrain",
        ],
        default="vit_small",
        help="Model architecture."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="Patch resolution of the model."
    )

    # artifact options
    parser.add_argument(
        "--check_artifacts",
        action="store_true",
        help="Enable artifact detection."
    )
    parser.add_argument(
        "--artifact_method",
        type=str,
        choices=["gap", "norm", "seed"],
        default="gap",
        help="Which method to use for artifact detection."
    )
    parser.add_argument(
        "--artifact_threshold",
        type=float,
        default=30.0,
        help="Threshold for deciding an artifact."
    )

    # input data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        default=None,
        help="Name of the dataset to run on (mutually exclusive with --image_path)."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If set, process only this single image file."
    )

    # where to save outputs / visualizations
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save any visualizations or logs."
    )
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    parser.add_argument(
        "--masked_artifact",
        action="store_true",
        help="Apply masked artifact filtering."
    )

    args = parser.parse_args()

    if args.image_path:
        dataset = ImageDataset(args.image_path)
    else:
        dataset = Dataset(args.dataset,dataset_set=args.set,remove_hards=args.no_hard)

    vis_folder = os.path.join(args.output_dir, "heatmaps")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, device)

    for im_id, inp in enumerate(dataset.dataloader):

        img = inp[0].to(device)
        im_name = dataset.get_image_name(inp[1])
        init_image_size = img.shape

        if im_name is None:
            continue

        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded
        #img = img.cuda(non_blocking=True)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        with torch.no_grad():
            out = model(img[None], is_training=True)
            features = out['x_norm_patchtokens']


        # 1) Artifact detection
        if args.check_artifacts:
            artifacts, id, norms, _, _, gap = detect_artifacts(
                features,
                method=args.artifact_method,
                threshold=args.artifact_threshold,
            ).values()
            if artifacts.any():
                print(f"[!] Artifact detected in {im_name}: indices {id}")


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

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import math 
import torch.nn.functional as F
from artifact_det import detect_artifacts
from datasets import ImageDataset, Dataset, bbox_iou
from networks import get_model
from object_discovery import lost, detect_box, dino_seg, compute_dynamic_k, compute_k_from_bboxes
from visualizations import visualize_fms, visualize_predictions, visualize_seed_expansion, visualize_heatmap
from count_patches_inside import *
from PIL import Image
from gate import AttentionReweightAndGate
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unsupervised object discovery with LOST.")
    parser.add_argument(
        "--arch",
        default="vit_small",
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
            "dinov2_vitb14_pretrain",
            "dinov2_vitl14_reg4_pretrain",
            "dinov2_vitg14_pretrain",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=14, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )
    parser.add_argument(
        "--artifact_remove", action="store_true", help="Remove artifacts from the image."
    )
    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["fms", "seed_expansion", "pred", None, "heatmap"],
        default=None,
        help="Select the different type of visualizations.",
    )

    # For ResNet dilation
    parser.add_argument("--resnet_dilate", type=int, default=2, help="Dilation level of the resnet model.")
    parser.add_argument("--threshold", type=int, default=10, help="Threshold for the dynamic thresholding.")
    parser.add_argument("--min_samples", type=int, default=2, help="Minimum number of samples for DBSCAN.")
    parser.add_argument("--eps", type=int, default=1, help="Epsilon for DBSCAN.")
    # LOST parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
    "--start_idx",
    type=int,
    default=0,
    help="Skip images before this index (zero-based) in the dataset",
)
#     parser.add_argument(
#         '--dynamic_k', 
#         action='store_true',
#         help='Enable dynamic k_patches based on image size')
#     parser.add_argument(
#     '--dynamic_k_bbox',
#     action='store_true',
#     help='Enable dynamic k_patches based on ground-truth bounding boxes'
# )

    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )

    parser.add_argument(
    "--dynamic_k",
    action="store_true",
    help="Compute k_patches on-the-fly from the size of the GT bounding box(es)."
)


    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    # Dynamic threshold
    parser.add_argument("--dynamic_thres", action="store_true", help="Use dynamic thresholding.")

    # masked artifact
    args = parser.parse_args()

    if args.dynamic_thres:
        print("Using dynamic thresholding.")

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, device)
    # wrap your ViT in our helper
    rewg = AttentionReweightAndGate(model, patch_size=args.patch_size).to(device)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with LOST
        exp_name = f"LOST-{args.arch}"
        if "resnet" in args.arch:
            exp_name += f"dilate{args.resnet_dilate}"
        elif "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"

    print(f"Running LOST on the dataset {dataset.name} (exp: {exp_name})")

    # Visualization
    if args.visualize:
        vis_folder = f"{args.output_dir}/visualizations/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))

    pbar = tqdm(dataset.dataloader)
    hard_ds = []
    all_blocks_out = []            # global list, reused each image

    # def make_hook():
    #     def fn(module, inp, out):
    #         # out shape: (B, N_tokens, D). keep batch-0, drop CLS
    #         all_blocks_out.append(out[0, 1:].detach())   # (N_patches, D)
    #     return fn

    # hooks = [blk.register_forward_hook(make_hook()) for blk in model.blocks]
    all_image_means = []
    total_semantic_images = 0
    total_k_less_than_5percent = 0
    total_k_less_than_20percent = 0
    total_k_less_than_5percent_potential = 0
    total_k_less_than_20percent_potential = 0

    start = args.start_idx
    for im_id, inp in enumerate(pbar):
        # ------------ IMAGE PROCESSING -------------------------------------------
        if im_id < start:
            continue
        all_blocks_out.clear()
        img = inp[0]
        init_image_size = img.shape
        image_size = init_image_size[1:3]  # (height, width)

# Override k_patches if dynamic mode is enabled
        
        
        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # Move to gpu

        #img = img.cuda(non_blocking=True)

        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size
        # ------------ GROUND-TRUTH -------------------------------------------
        gt_bbxs = None   
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)
        #     print("bbox", gt_bbxs)
        #     if args.dynamic_k:  # Add this flag to your argument parser
        #         args.k_patches = compute_dynamic_k(image_size)
        #         print("k patch size ", args.k_patches)
        #     elif args.dynamic_k_bbox and not args.no_evaluation and gt_bbxs is not None:
        #         args.k_patches = compute_k_from_bboxes(gt_bbxs, image_size)
        #         print(f"[Dynamic K from BBox] k_patches set to {args.k_patches}")
        #     print("image shape", init_image_size)
        #     if gt_bbxs is not None:
        #         # Discard images with no gt annotations
        #         # Happens only in the case of VOC07 and VOC12
        #         if gt_bbxs.shape[0] == 0 and args.no_hard:
        #             continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit" in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                # ==================================
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                attentions = model.get_last_selfattention(img[None, :, :, :])
                print("attn shape: ", attentions.shape)
                # Scaling factor
                scales = [args.patch_size, args.patch_size]

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                if args.dinoseg:
                    pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                    pred = np.asarray(pred)
                else:

                    attn = attentions[0, :, 0, 1:].reshape(nh,-1)
                    
                    if args.artifact_remove:

                        def det_artifact(attn):
                            max_indices = torch.argmax(attn, dim=1)  # shape: (nh,)
                            counts = torch.bincount(max_indices)
                            idx = torch.argmax(counts)
                            return idx.item()

                        art_id = det_artifact(attn)
                        attn[:,art_id] = 0
                    attn = attn.reshape(nh, w_featmap, h_featmap)
                    attn = nn.functional.interpolate(attn.unsqueeze(0),
                                                     scale_factor=args.patch_size,
                                                     mode='nearest')[0].cpu().numpy()
                    qkv = (
                        feat_out["qkv"]
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                   

                    q_cls = q[:, :, 0, :]  #
                    q_all = q[:, :, 1:, :]
                    print(q_all.shape)
                    print(q_cls.shape)
                    attn1 = torch.einsum("bhd,bhnd->bhn", q_cls, q_all)  # (1, 6, 1008)
                    print(attn1.shape)
                    attn1 = attn1 / (64**0.5)
                    attn1 = torch.softmax(attn1, dim=-1).squeeze()
                    attn1 = attn1.reshape(nh, w_featmap, h_featmap)
                    attn1 = nn.functional.interpolate(attn1.unsqueeze(0),
                                                        scale_factor=args.patch_size,
                                                        mode='nearest')[0].cpu().numpy()

                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

                    feats1 = None
                    if args.which_features == "k":
                        feats = k[:, 1:, :]
                        #feats1 = q[:, 1:, :]
                    elif args.which_features == "q":
                        feats = q[:, 1:, :]
                    elif args.which_features == "v":
                        feats = v[:, 1:, :]
                    if 'reg' in args.arch:
                        feats = feats[:, 4:, :]

        # ------------ Apply LOST -------------------------------------------
        if not args.dinoseg:
            # Apply LOST
            # The larger bboxes, the less patches
            def k_from_bboxes(gt_boxes, img_h, img_w, patch_size,
                  scale_factor=0.5, k_min=70, k_max=500):
                """
                Estimate k (number of lowest-degree patches) from GT boxes.
                - gt_boxes:  (N,4)  ndarray | tensor  [x0,y0,x1,y1] in pixels
                - img_h, img_w:      original image size in pixels
                - patch_size:        ViT patch edge length in pixels
                Returns an integer between k_min and k_max.
                """
                if gt_boxes is None or len(gt_boxes) == 0:
                    return k_min                        # fallback when no GT
                # total GT area
                areas = (gt_boxes[:,2] - gt_boxes[:,0]) * (gt_boxes[:,3] - gt_boxes[:,1])
                gt_area   = areas.sum()
                img_area  = img_h * img_w
                if gt_area > img_area:
                    return k_min  # fallback when GT area is larger than image area
                # fraction of the image covered by all objects
                frac = float(gt_area) / (img_area + 1e-6)

                inv_frac = 1.0 - min(max(frac, 0.0), 1.0)

                # total number of patches
                n_rows = math.ceil(img_h / patch_size)
                n_cols = math.ceil(img_w / patch_size)
                tot_patches = n_rows * n_cols

                # estimate and clamp
                k_est = int(inv_frac * scale_factor * tot_patches)
                k_clamped = max(k_min, min(k_est, k_max))
                return k_clamped

            # The larger bboxes, the more patches
            # def k_from_bboxes(gt_boxes, img_h, img_w, patch_size,
            #       scale_factor=0.5, k_min=70, k_max=500):
            #     """
            #     Estimate k (number of lowest-degree patches) from GT boxes,
            #     so that larger GT area produces a larger k.

            #     - gt_boxes:  (N,4) ndarray | tensor [x0,y0,x1,y1] in pixels
            #     - img_h, img_w: original image size
            #     - patch_size:   ViT patch edge length
            #     Returns k between k_min and k_max.
            #     """
            #     # fallback when no GT
            #     if gt_boxes is None or len(gt_boxes) == 0:
            #         return k_min

            #     # total GT area
            #     areas = (gt_boxes[:,2] - gt_boxes[:,0]) * (gt_boxes[:,3] - gt_boxes[:,1])
            #     gt_area  = areas.sum()
            #     img_area = img_h * img_w

            #     # fraction of the image covered by all objects
            #     frac = float(gt_area) / (img_area + 1e-6)
            #     # clamp to [0,1]
            #     frac = min(max(frac, 0.0), 1.0)

            #     # total number of patches
            #     n_rows     = math.ceil(img_h / patch_size)
            #     n_cols     = math.ceil(img_w / patch_size)
            #     tot_patches = n_rows * n_cols

            #     # now larger frac → larger k_est
            #     k_est = int(frac * scale_factor * tot_patches)

            #     # clamp into [k_min, k_max]
            #     k_clamped = max(k_min, min(k_est, k_max))
            #     return k_clamped
            if args.dynamic_k:
                cur_k = k_from_bboxes(
                    gt_bbxs,                         # can be None in inference mode
                    init_image_size[1],              # height
                    init_image_size[2],              # width
                    args.patch_size
                )
            else:
                cur_k = args.k_patches
            print(f"Using k={cur_k} patches for {im_name}")
            img_tensor = img[None].to(device)  # shape (1,C,H,W)

            # ▶️ Step 1+2 fused attention scores:
            print("img_tensor shape", img_tensor.shape)
            fused_scores = rewg(img_tensor)  # shape (1, N_tokens)
            pred, A, scores, seed, potentials, jumps = lost(
                feats,
                feats1 if feats1 is not None else feats,
                [w_featmap, h_featmap],
                scales,
                init_image_size,
                k_patches=cur_k,
                dynamic_thres="dinov2" in args.arch,
                artifact_idx = art_id if args.artifact_remove else None,
                custom_scores=fused_scores
            )
            VOC12_SEG_DIR = "datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass"
            
            mask_path = os.path.join(VOC12_SEG_DIR, im_name.replace('.jpg','.png'))
            if not os.path.isfile(mask_path):
                print(f"[{im_name}] no segmentation mask found, skipping image")
                continue
            total_semantic_images += 1
            mask = np.array(Image.open(mask_path))
            # assume mask pixels > 0 are “semantic object”
            mask = (mask > 0).astype(np.uint8)

            total, in_k = count_semantic_patches(
                potentials,
                mask,
                args.patch_size,
                w_featmap,
                h_featmap
            )

            print(f"{im_name}:")
            print(f"  patches covering object region: {total}/{w_featmap*h_featmap}")
            print(f"  of those, {in_k}/{len(potentials)} are among LOST patches")
            frac = (in_k / total) * 100 if total else 0 # number of k_patches / total)number of semantics (area of cover)
            if frac > 30:
                total_k_less_than_5percent += 1
            if frac > 50:
                total_k_less_than_20percent += 1

            frac_potential = (in_k / len(potentials)) * 100 if len(potentials) else 0 # num of k patches / total potentials (accuracy)
            if frac_potential > 30:
                total_k_less_than_5percent_potential += 1
            if frac_potential > 50:
                total_k_less_than_20percent_potential += 1
            if total_semantic_images:
                print(f"→ {total_k_less_than_5percent}/{total_semantic_images} images had >30% of patches covering semantic object")
                print(f"{total_k_less_than_20percent}/{total_semantic_images} images had >50% of patches covering semantic object")
                print(f"→ {total_k_less_than_5percent_potential}/{total_semantic_images} images had >30% of potentials LOST patches covering semantic object")
                print(f"→ {total_k_less_than_20percent_potential}/{total_semantic_images} images had >50% of potentials LOST patches covering semantic object")
            
            # ------------------------------------------------------------------
            # 2.  inside the image loop, AFTER you have `potentials`
            # ------------------------------------------------------------------
           # ------------------------------------------------------------------
# forward once so hooks collect the per-block outputs
# ------------------------------------------------------------------
        #     all_blocks_out.clear()
        #     ### anomoly detection
        #     from singular_defect import anomaly_dir_attn, anomaly_dir_mlp_ls, singular_defect_directions
        #    #anamolies = singular_defect_directions(model)
        #     _ = model.forward_features(img[None].to(device))

        #     has_reg = "dinov2" in args.arch or "reg" in args.arch
        #     N_REG   = 4 if has_reg else 0

        #     patch_idx_full = torch.as_tensor(potentials,
        #                                     device=all_blocks_out[0].device) + N_REG


        #     angles_evo, angles_ref0 = [], []

        #     for b in range(1, len(all_blocks_out)):
        #         tokens_b = all_blocks_out[b]                    # (T_b, D)
        #         tokens_prev = all_blocks_out[b-1]               # (T_{b-1}, D)

        #         # keep only the indices that exist in this block
        #         mask = (patch_idx_full < tokens_b.size(0)) & \
        #         (patch_idx_full < tokens_prev.size(0))
        #         keep = patch_idx_full[mask]
                
        #         if keep.numel() == 0:
        #             continue                                    # no common tokens

        #         cur  = tokens_b[keep]                           # (k', D)
        #         prev = tokens_prev[keep]                        # (k', D)

        #         cos  = F.cosine_similarity(cur, prev, dim=1).clamp(-1+1e-7, 1-1e-7)
        #         ang  = torch.acos(cos) * 180.0 / math.pi
        #         angles_evo.append(ang.cpu())

        #     # ---------- optional: drift w.r.t. first block ---------------------
        #     ref0 = all_blocks_out[0]
        #     keep0 = patch_idx_full[patch_idx_full < ref0.size(0)]
        #     ref0_sel = ref0[keep0]

        #     for b in range(1, len(all_blocks_out)):
        #         tb = all_blocks_out[b]
        #         keep = patch_idx_full[patch_idx_full < tb.size(0)]
        #         if keep.numel() == 0:
        #             continue

        #         cur = tb[keep]
        #         ref = ref0_sel[: keep.numel()]                  # align lengths
        #         cos = F.cosine_similarity(cur, ref, dim=1).clamp(-1+1e-7, 1-1e-7)
        #         ang = torch.acos(cos) * 180.0 / math.pi
        #         angles_ref0.append(ang.cpu())

        #     if angles_evo:
        #         print(f"[{im_name}] mean per-block turn = "
        #             f"{torch.cat(angles_evo).mean():.1f}°  "
        #             f"final drift = {torch.cat(angles_ref0).mean():.1f}°")
        #     else:
        #         print(f"[{im_name}]  no patch survives token-merging layers")

        #     print("\n=====  θ(k-patch) between successive blocks  =====")
        #     for b, ang in enumerate(angles_evo, start=1):          # b = 1 … L-1
        #         # ang is a 1-D tensor of length k′ (might be < k if some patches skipped)
        #         vals = ", ".join(f"{x:.2f}" for x in ang.tolist())
        #         print(f"block {b-1:02d} → {b:02d}:  [{vals}]  (mean={ang.mean():.2f}°)")

        #     print("\n=====  θ(k-patch) vs. first block (drift)  =====")
        #     for b, ang in enumerate(angles_ref0, start=1):
        #         vals = ", ".join(f"{x:.2f}" for x in ang.tolist())
        #         print(f"block 00 → {b:02d}:  [{vals}]  (mean={ang.mean():.2f}°)")
            
        #     print("")

        #     k = patch_idx_full.numel()
        #     pairwise_stats = []
        #     for b, tokens_b in enumerate(all_blocks_out):
        #         valid = patch_idx_full < tokens_b.size(0)
        #         if not valid.any():
        #             # none of the k patches survived in this block
        #             pairwise_stats.append(float('nan'))
        #             continue

        #         keep = patch_idx_full[valid]           # (k_b,)
        #         sel  = tokens_b[keep]             # (k_b, D)


        #         # pair-wise cosine similarities
        #         cos = F.cosine_similarity(
        #                 sel.unsqueeze(1),      # (k,1,D)
        #                 sel.unsqueeze(0),      # (1,k,D)
        #                 dim=-1)                # -> (k,k)

        #         # mask out the diagonal
        #         off_diag = cos[~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)]

        #         # convert to angles (radians → degrees)
        #         angles = torch.acos(off_diag.clamp(-1+1e-7, 1-1e-7)) * (180.0 / math.pi)

        #         # average angle
        #         mean_angle = angles.mean().item()
        #         pairwise_stats.append(mean_angle)
            # print('Per-block mean off-diagonal *angle*:')
            # for b, ang in enumerate(pairwise_stats):
            #     print(f'  block {b:02d}:  {ang:6.2f}°')

            # ------Count patches in GT box-----------------------------------------
            
            # per_box_counts = []
            # outside_cnt    = 0
            # if (not args.no_evaluation) and (gt_bbxs is not None):
            #     per_box_counts, outside_cnt = potentials_in_boxes(
            #                                 potentials, w_featmap, h_featmap,
            #                                 args.patch_size, gt_bbxs)
            #     print(f"Potentials per GT box  : {per_box_counts}")
            #     print(f"Potentials outside all : {outside_cnt} / {len(potentials)}")
            #     if per_box_counts:
            #         mean_per_box = sum(per_box_counts) / len(per_box_counts)
            #     else:
            #         mean_per_box = 0.0
            #     print(f"Mean patches inside per GT box: {mean_per_box:.2f}")

            #     all_image_means.append(mean_per_box)

            # ------------ Visualizations -------------------------------------------
            if args.visualize == "fms":
                visualize_fms(A.clone().cpu().numpy(), seed, scores, [w_featmap, h_featmap], scales, vis_folder, im_name)

            elif args.visualize == "seed_expansion":
                image = dataset.load_image(im_name)

                # Before expansion
                pred_seed, _ = detect_box(
                    A[seed, :],
                    seed,
                    [w_featmap, h_featmap],
                    scales=scales,
                )
                visualize_seed_expansion(image, pred, seed, pred_seed, scales, [w_featmap, h_featmap], vis_folder, im_name)

            elif args.visualize == "pred":
                image = dataset.load_image(im_name)
                #visualize_predictions(image, pred, seed, scales, [w_featmap, h_featmap], vis_folder, im_name, plot_seed=True, potentials=potentials, char=args.which_features, perbox_counts=per_box_counts, outside_counts=outside_cnt, gt_bboxes=gt_bbxs)
                visualize_predictions(image, pred, seed, scales, [w_featmap, h_featmap], vis_folder, im_name, plot_seed=True, potentials=potentials, char=args.which_features, perbox_counts=in_k, gt_bboxes=gt_bbxs, mask=mask)

                
            elif args.visualize == "heatmap":
                visualize_heatmap(attn1, im_name, vis_folder,mean=True)
            
            del A, scores, attn, qkv, feats, feats1, potentials, jumps

        # Save the prediction
        preds_dict[im_name] = pred

        # Evaluation
        if args.no_evaluation:
            continue
        # convert pred to numpy
        pred = np.array(pred)

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}, is_artifact: {len(hard_ds)}")

    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
        print('File saved at %s'%result_file)
    # if all_image_means:
    #     dataset_mean = sum(all_image_means) / len(all_image_means)
    #     print(f"\n→ Dataset-wide mean patches inside per GT box: {dataset_mean:.2f}")
    if total_semantic_images:
        print(f"→ {total_k_less_than_5percent}/{total_semantic_images} images had <5% of patches covering semantic object")
        print(f"{total_k_less_than_20percent}/{total_semantic_images} images had <20% of patches covering semantic object")
        print(f"→ {total_k_less_than_5percent_potential}/{total_semantic_images} images had <5% of potentials LOST patches covering semantic object")
        print(f"→ {total_k_less_than_20percent_potential}/{total_semantic_images} images had <20% of potentials LOST patches covering semantic object")
            
    
    # for h in hooks:
    #     h.remove()
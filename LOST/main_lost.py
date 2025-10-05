
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
from dataset import ImageDataset, Dataset, bbox_iou
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
    parser.add_argument(
        "--artifact_detect", action="store_true", help = "Verify that if artifact is sink"
    )
    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["fms", "seed_expansion", "pred", None, "heatmap", "svd"],
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
    parser.add_argument(
    "--use_sinder", 
    action="store_true", 
    help="Use Sinder for artifact detection."
)
    parser.add_argument(
    "--detect_sinks",
    action="store_true",
    help="Detect and mask attention sink tokens (as in 'What are you sinking?')."
)


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
    print("Using device:", device)
    model = get_model(args.arch, args.patch_size, device)
    from pathlib import Path
    import sys
    from pathlib import Path

    # add SINDER repo to sys.path
    sinder_path = "/Users/tringuyen/Documents/Multimodal_Research/object_localiztation/sinder"
    if sinder_path not in sys.path:
        sys.path.append(sinder_path)

    # now safe to import
    from sinder.utils import load_model
    from sinder.singular_defect import singular_defect_directions
    import sinder
    def load_model(model_name, checkpoint=None):
        print(f'using {model_name} model')
        model = torch.hub.load(
            repo_or_dir=Path(sinder.__file__).parent.parent,
            source='local',
            model=model_name,
        )
        if checkpoint is not None:
            states = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(states, strict=False)
        #model = model.cuda()
        model.eval()
        model.interpolate_antialias = True
        model.singular_defects = singular_defect_directions(model)
        print(f'model loaded. patch size: {model.patch_size}')


        return model
    #model = load_model('dinov2_vitg14')
    import torch.nn.functional as F

    # def get_last_selfattention(self, x):
    #     """
    #     Forward once and grab last block's attention weights.
    #     """
    #     attn_weights = None

    #     def hook_fn(module, input, output):
    #         nonlocal attn_weights
    #         attn_weights = module.attn_drop(output)  # or just output if drop not applied

    #     # hook into the last attention layer
    #     handle = self.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

    #     # forward pass
    #     _ = self.forward(x)

    #     handle.remove()
    #     return attn_weights

    # # Attach to model dynamically
    # setattr(model, "get_last_selfattention", get_last_selfattention.__get__(model))

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

    feat_out_id = {}

    def make_hook(layer_id):
        def hook_fn(module, input, output):
            feat_out_id[layer_id] = output.detach()  # store qkv for this layer
        return hook_fn
    layer_attns = {}

    def make_attn_hook(layer_id):
        def hook_fn(module, input, output):
            # output here is attn probabilities (after softmax)
            # shape: (B, H, N, N)
            layer_attns[layer_id] = output.detach().cpu()
        return hook_fn
    for i, blk in enumerate(model.blocks):
        blk.attn.qkv.register_forward_hook(make_hook(i))
        blk.attn.attn_drop.register_forward_hook(make_attn_hook(i))

    all_image_means = []
    total_semantic_images = 0
    total_k_less_than_5percent = 0
    total_k_less_than_20percent = 0
    total_k_less_than_5percent_potential = 0
    total_k_less_than_20percent_potential = 0

    start = args.start_idx
    all_image_means = []
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

                # # Forward pass in the model
                attentions = model.get_last_selfattention(img[None, :, :, :])
               # print("attn shape: ", attentions.shape)
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
                    import torch
                    import torch.nn.functional as F

                    

                    # Put it back later
                    # attn = attn.reshape(nh, w_featmap, h_featmap)
                    # attn = nn.functional.interpolate(attn.unsqueeze(0),
                    #                                  scale_factor=args.patch_size,
                    #                                  mode='nearest')[0].cpu().numpy()


                    qkv = (
                        feat_out["qkv"]
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                   

                    q_cls = q[:, :, 0, :]  #
                    q_all = q[:, :, 1:, :]
                    k_cls = k[:, :, 0, :]  
                    k_all = k[:, :, 1:, :]

                   # q_cls = F.normalize(q_cls, p=2, dim=-1, eps=1e-8)   # (B, H, D), normalized
                   # k_all = F.normalize(k_all, p=2, dim=-1, eps=1e-8)  
                   
                    print(q_all.shape)
                    print(q_cls.shape)
                    #attn1 = torch.einsum("bhd ,bhnd->bhn", q_cls, k_all)  # (1, 6, 1008)
                    
                    #attn1 = torch.einsum("bhd,bhd->hd", q_cls, q_cls)
                    # print("att shape", attn.shape)
                    # print("attn1 shape", attn1.shape)

                    # attn1 = attn1 / (64**0.5)

                    #attn1 = torch.softmax(attn1, dim=-1).squeeze()
                    
                    attn2 = torch.matmul(q_cls.unsqueeze(-2), k_all.transpose(-2,-1)).squeeze(-2).squeeze(0)
                    attn2 = attn2 / (math.sqrt(q_cls.size(-1)))
                    def apply_sinder_artifact_detection(model, attn, w_featmap, h_featmap, mask_thr=4, skip_less_than=3):
                        
                        print("Computing Sinder singular defect directions...")
                        anomalies = singular_defect_directions(model)

                        if anomalies and len(anomalies) > 0:
                            final_anomaly = anomalies[-1] if isinstance(anomalies, list) else anomalies
                            num_patches = w_featmap * h_featmap
                            if len(final_anomaly) > num_patches:
                                final_anomaly = final_anomaly[:num_patches]
                            elif len(final_anomaly) < num_patches:
                                padding = torch.zeros(num_patches - len(final_anomaly), device=final_anomaly.device)
                                final_anomaly = torch.cat([final_anomaly, padding])

                            anomaly_scores = torch.abs(final_anomaly)

                            # Threshold using mean + mask_thr * std
                            mean, std = anomaly_scores.mean(), anomaly_scores.std()
                            threshold_val = mean + mask_thr * std
                            artifact_mask = anomaly_scores > threshold_val
                            artifact_indices = torch.where(artifact_mask)[0]

                            # Apply skip_less_than
                            if len(artifact_indices) < skip_less_than:
                                print(f"Too few artifacts ({len(artifact_indices)}), skipping")
                                return []

                            print(f"Sinder detected {len(artifact_indices)} artifacts with mask_thr={mask_thr}")
                            print(f"Artifact indices: {artifact_indices.tolist()}")

                            return artifact_indices.tolist()
                        else:
                            print("No anomalies detected by Sinder")
                            return []

                        
                        
                                
                    if args.use_sinder:
                        # Use Sinder for artifact detection
                        artifact_indices = apply_sinder_artifact_detection(
                            model, attn, w_featmap, h_featmap
                        )
                        
                        # Remove artifacts from attention
                        for art_id in artifact_indices:
                            if art_id < attn.shape[1]:  # Ensure index is valid
                                attn[:, art_id] = 0.0
                    if args.artifact_remove:
                        def det_artifact(attn_2d: torch.Tensor) -> int:
                            """
                            attn_2d: (H, N) attention per head.
                            Returns the token index that most heads spike on (the artifact column).
                            """
                            # index of max token per head
                            max_indices = torch.argmax(attn_2d, dim=1)  # (H,)
                            # count how many heads pick each token
                            counts = torch.bincount(max_indices, minlength=attn_2d.size(1))  # (N,)
                            return int(torch.argmax(counts))
                        

                        art_id = det_artifact(attn1)
                        attn1[:, art_id] = 0.0
                        attn1 = attn1 / (attn1.sum(dim=1, keepdim=True) + 1e-8)
                    def detect_attention_sinks(attn: torch.Tensor, tau=0.9, gamma=0.3):
                        """
                        Detect attention sink tokens.

                        Args:
                            attn: (H, N) tensor of attention weights per head (already softmaxed).
                            tau: percentile threshold (e.g., top 10%).
                            gamma: fraction of heads that must spike on the same token.

                        Returns:
                            sink_indices: list of token indices considered sinks.
                        """
                        H, N = attn.shape
                        sink_counts = torch.zeros(N, device=attn.device)

                        # For each head, count tokens above tau-percentile
                        for h in range(H):
                            thresh = torch.quantile(attn[h], tau)
                            mask = attn[h] >= thresh
                            sink_counts += mask.float()

                        # Normalize to [0,1]
                        sink_freq = sink_counts / H

                        # Tokens where ≥ gamma fraction of heads spiked
                        sink_indices = torch.where(sink_freq >= gamma)[0]

                        return sink_indices.tolist()
                    
                    if args.detect_sinks:
                        sink_indices = detect_attention_sinks(attn1, tau=0.99, gamma=1)
                        print(f"Detected attention sinks: {sink_indices}")

                        # Zero out sinks if you want to remove their influence
                        for sink_id in sink_indices:
                            print(f"Masking sink token {sink_id} with attention sum {attn1[:, sink_id].sum().item():.4f}")
                            attn1[:, sink_id] = 0.0

                        # Renormalize
                        attn1 = attn1 / (attn1.sum(dim=1, keepdim=True) + 1e-8)
                    import matplotlib.pyplot as plt
                    if args.artifact_detect:
                        def save_heatmap(arr, save_path, title=""):
                            """
                            Save a 2D NumPy array as a heatmap.
                            """
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            plt.figure(figsize=(5, 5))
                            plt.imshow(arr, cmap="viridis")
                            plt.colorbar()
                            plt.title(title)
                            plt.tight_layout()
                            plt.savefig(save_path)
                            plt.close()
                        
                        def detect_artifact_and_divergence(attn: torch.Tensor):
                            """
                            Detect artifact sink token and compute KL divergence before/after masking.

                            Args:
                                attn: (H, N) tensor of attention logits or weights per head.
                                    H = #heads, N = #tokens.

                            Returns:
                                artifact_idx: int, index of detected artifact token
                                agreement: float, fraction of heads that spike on artifact_idx
                                stats: dict with per-head KL values and reductions
                            """
                            H, N = attn.shape

                            # 1. Convert to probabilities
                            p_all = F.softmax(attn, dim=-1)  # (H, N)

                            # 2. Detect artifact index (majority vote of argmax)
                            max_indices = torch.argmax(p_all, dim=1)  # (H,)
                            counts = torch.bincount(max_indices, minlength=N)
                            artifact_idx = torch.argmax(counts).item()
                            agreement = counts[artifact_idx].item() / H
                            divergences = []
                            # 3. KL computations
                            stats = {"KL_original": [], "KL_without_sinks": [], "KL_reduction": []}
                            u = torch.full((N,), 1.0 / N, device=attn.device)  # uniform baseline

                            for h in range(H):
                                p = p_all[h].clamp(min=1e-8)  # original probs

                                # ---- KL(original vs uniform)
                                kl_orig = F.kl_div(u.log(), p, reduction="sum").item()

                                # ---- mask artifact token, re-softmax
                                masked = attn[h].clone()
                                masked[artifact_idx] = -1e9  # mask sink
                                q = F.softmax(masked, dim=0).clamp(min=1e-8)

                                # ---- KL(without sinks vs uniform)
                                kl_nosink = F.kl_div(u.log(), q, reduction="sum").item()

                                # ---- Reduction
                                kl_reduction = kl_orig - kl_nosink
                                kl = F.kl_div(q.log(), p, reduction="sum").item()
                                divergences.append(kl)
                                stats["KL_original"].append(kl_orig)
                                stats["KL_without_sinks"].append(kl_nosink)
                                stats["KL_reduction"].append(kl_reduction)

                            return divergences, stats

                        def detect_artifact_and_kl_reduction(attn: torch.Tensor):
                            """
                            Detect sink token and compute KL reduction per head.

                            Args:
                                attn: (H, N) tensor of attention logits (not normalized).
                                    H = #heads, N = #tokens.

                            Returns:
                                artifact_idx: int, index of detected artifact token
                                agreement: float, fraction of heads that spike on artifact_idx
                                kl_to_avg: list of KL(p_h || avg_p) per head
                                kl_without_sinks: list of KL(q_h || avg_q) per head
                                kl_reduction: list of KL reduction values per head
                            """
                            H, N = attn.shape

                            # 1. Normalize attention
                            p_all = F.softmax(attn, dim=1)  # (H, N)

                            # 2. Detect artifact by majority argmax
                            max_indices = torch.argmax(p_all, dim=1)
                            counts = torch.bincount(max_indices, minlength=N)
                            artifact_idx = torch.argmax(counts).item()
                            agreement = counts[artifact_idx].item() / H

                            # 3. Mask artifact → renormalize
                            masked_logits = attn.clone()
                            masked_logits[:, artifact_idx] = 1e-8
                            q_all = F.softmax(masked_logits, dim=1)  # (H, N)
                            
                            # 4. Average distributions
                            avg_p = p_all.mean(dim=0)  # (N,)
                            avg_q = q_all.mean(dim=0)  # (N,)

                            # 5. Compute KLs
                            kl_to_avg, kl_without_sinks, kl_reduction = [], [], []
                            for h in range(H):
                                # p = p_all[h]
                                # q = q_all[h]
                                eps = 1e-8
                                p = p_all[h].clamp(min=eps)      # (N,)
                                q = q_all[h].clamp(min=eps)      # (N,)
                                avg_p = avg_p.clamp(min=eps)
                                avg_q = avg_q.clamp(min=eps)


                                # KL(p_h || avg_p)
                                kl1 = F.kl_div( p.log(),avg_p, reduction="sum").item()
                                # KL(q_h || avg_q)
                                kl2 = F.kl_div(q.log(),avg_q, reduction="sum").item()

                                kl_to_avg.append(kl1)
                                kl_without_sinks.append(kl2)
                                kl_reduction.append(kl1 - kl2)
                                print(f"Head {h} \n KL to avg: {kl1:.4f}")
                                print(f"KL without sinks: {kl2:.4f}")
                                print(f"KL Reduction: {kl1 - kl2:.4f}")
                            avg_kl_to_avg = float(sum(kl_to_avg) / H)
                            avg_kl_without_sinks = float(sum(kl_without_sinks) / H)
                            avg_kl_reduction = float(sum(kl_reduction) / H)
                            print("Average KL to avg:", avg_kl_to_avg)
                            print("Average KL without sinks:", avg_kl_without_sinks)
                            print("Average KL reduction:", avg_kl_reduction)
                            return {
                                "artifact_idx": artifact_idx,
                                "agreement": float(agreement),
                                "kl_to_avg": kl_to_avg,
                                "kl_without_sinks": kl_without_sinks,
                                "kl_reduction": kl_reduction,

                            }
                        # def detect_artifact_and_divergence(attn: torch.Tensor):
                        #     """
                        #     Detect artifact sink token and compute KL divergence before/after masking.
                            
                        #     Args:
                        #         attn: (H, N) tensor of attention weights per head (softmaxed).
                        #             H = #heads, N = #tokens.
                            
                        #     Returns:
                        #         artifact_idx: int, index of detected artifact token
                        #         agreement: float, fraction of heads that spike on artifact_idx
                        #         divergences: list of KL divergence values, one per head
                        #     """
                        #     H, N = attn.shape

                        #     # 1. Detect artifact index based on majority argmax
                            
                        #     p_all = F.softmax(attn, dim=-1)  # (H, N)
                        #     #p_all = attn 
                        #     # 1. Detect artifact index based on majority argmax
                        #     max_indices = torch.argmax(p_all, dim=1)  # (H,)
                        #     counts = torch.bincount(max_indices, minlength=N)
                        #     artifact_idx = torch.argmax(counts).item()
                        #     agreement = counts[artifact_idx].item() / H

                        #     # 2. Compute KL divergence per head (original vs. masked)
                        #     divergences = []
                        #     delta_matrices = torch.zeros_like(p_all)
                            


                        #     for h in range(H):
                        #         p = p_all[h]  # already softmaxed (N,)

                        #         # Mask artifact index, then softmax again
                        #         masked = attn[h].clone()              # raw row (logits or probs input)
                        #         masked[artifact_idx] = 1e-8  # remove sink token
                        #         q = F.softmax(masked, dim=0)          # re-softmax (N,)

                        #         # KL(p || q)
                        #         kl = F.kl_div(q.log(), p, reduction="sum").item()
                        #         divergences.append(kl)
                        #         delta_matrices[h] = p - q

                        #     print(divergences)
                        #     return divergences,delta_matrices
                        


                        # def detect_artifact_and_divergence(attn: torch.Tensor):
                        #     """
                        #     Detect artifact sink token and compute KL divergence before/after masking.
                            
                        #     Args:
                        #         attn: (H, N) tensor of attention weights per head (softmaxed).
                        #             H = #heads, N = #tokens.
                            
                        #     Returns:
                        #         artifact_idx: int, index of detected artifact token
                        #         agreement: float, fraction of heads that spike on artifact_idx
                        #         divergences: list of KL divergence values, one per head
                        #     """
                        #     H, N = attn.shape
                        #     attn = F.softmax(attn, dim=1)  # (H, N)
                        #     # 1. Detect artifact index based on majority argmax
                        #     max_indices = torch.argmax(attn, dim=1)  # (H,)
                        #     counts = torch.bincount(max_indices, minlength=N)
                        #     artifact_idx = torch.argmax(counts).item()
                        #     agreement = counts[artifact_idx].item() / H

                        #     # 2. Compute KL divergence per head (original vs. masked)
                        #     divergences = []
                        #     for h in range(H):
                        #         p = attn[h]  # (N,)
                        #         p = p / (p.sum() + 1e-8)  # normalize to prob
                        #         eps = 1e-8
                        #         # Mask artifact index
                        #         q = p.clone()
                        #         q[artifact_idx] = 1e-8
                        #         q = q / (q.sum() + 1e-8)  # renormalize
                        #         #p = torch.softmax(p, dim=-1).squeeze()
                        #         #q = torch.softmax(q, dim=-1).squeeze()
                        #         p = p.clamp(min=eps)
                        #         q = q.clamp(min=eps)
                        #         # KL(p || q)
                        #         kl = F.kl_div(q.log(), p, reduction="sum").item()
                        #         divergences.append(kl)

                        #         before = attn[h].view(w_featmap, h_featmap)
                        #         after  = q.view(w_featmap, h_featmap)

                        #         base_name = im_name.replace(".jpg", f"_head{h:02d}")
                        #         save_heatmap(before, os.path.join(vis_folder ,"heatmap", base_name + "_before.png"),
                        #                     title=f"Head {h} Before (KL={kl:.4f})")
                        #         save_heatmap(after, os.path.join(vis_folder , "heatmap", base_name + "_after.png"),
                        #                     title=f"Head {h} After (KL={kl:.4f})")

                        #     return artifact_idx, agreement, divergences
                        #attn1 = attn1.squeeze(0)
                        #artifact_idx, agreement, divergences = detect_artifact_and_divergence(attn1)
                        
                        #divergences, delta_matrices = detect_artifact_and_divergence(attn2)

                        # divergences, stats = detect_artifact_and_divergence(attn2)
                        # print("mean KL original:", np.mean(stats["KL_original"]))
                        # print("mean KL reduction:", np.mean(stats["KL_reduction"]))
                        
                        #  for i, d in enumerate(divergences):
                        #     print(f"Head {i}: KL divergence = {d:.4f}")
                        # # Print mean across images
                        # all_image_means.append(np.mean(divergences))
                        # mean_divergence = np.mean(all_image_means)
                        # print("current image mean divergence:", np.mean(divergences))
                        # print(f"Mean KL divergence across {len(all_image_means)} images: {mean_divergence:.4f}")
                        

                        
                        # stats = detect_artifact_and_kl_reduction(attn2)
                        # all_image_means.append(np.mean(stats["kl_reduction"]))
                        # mean_kl_reduction = np.mean(all_image_means)
                        # print("Mean KL reduction this image:", mean_kl_reduction)

                        
                        # kl_results = {}

                        # for layer_id, attn_probs in layer_attns.items():
                        #     # attn_probs: (B,H,N,N)
                        #     attn_new = attn_probs[0,:,0,1:]   # e.g. CLS→tokens, shape (H,N-1)
                            
                        #     stats = detect_artifact_and_kl_reduction(attn_new)  # your function
                        #     kl_results[layer_id] = {
                        #         "artifact_idx": stats["artifact_idx"],
                        #         "agreement": stats["agreement"],
                        #         "avg_kl_reduction": float(np.mean(stats["kl_reduction"]))
                        #     }
                        # layerwise_means = {i: [] for i in range(len(model.blocks))}


                        # for layer_id, stats in kl_results.items():
                        #     layerwise_means[layer_id].append(stats["avg_kl_reduction"])
                        #     print(f"Layer Avg KL reduction = {stats['avg_kl_reduction']:.4f}")

                        # after all images:
                        # for layer_id, vals in layerwise_means.items():
                        #     mean_val = np.mean(vals)
                        #     print(f"Layer {layer_id}: dataset mean KL reduction = {mean_val:.4f}")


                        
                       

                    #attn1 = torch.softmax(attn1, dim=-1).squeeze()


#                    attn1 = attn1.reshape(nh, w_featmap, h_featmap)
#                   attn1 = nn.functional.interpolate(attn1.unsqueeze(0),
#                                                        scale_factor=args.patch_size,
#                                                       mode='nearest')[0].cpu().numpy()



                    # B, H, N, D = q_all.shape                     # (1, 16, 1152, 64)

                    # sim = torch.einsum('b h n d, b h m d -> b h n m', q_all, q_all)
                    # sim = sim / math.sqrt(D)                      # scale
                    # attn1 = sim.softmax(dim=-1)                   # (B, H, N, N)

                    # # ---- collapse the last dimension ----
                    # attn1 = attn1.mean(dim=-1)                    # → (B, H, N) ❶
                    # #  or: attn1 = attn1.max(dim=-1).values      #   (choose the reduction you want)

                    # attn1 = attn1.squeeze(0)                      # → (H, N)     ❷
                    # attn1 = attn1.reshape(H, w_featmap, h_featmap)# → (16, 32, 36)

                    # attn1 = F.interpolate(attn1.unsqueeze(0),     # upscale to pixel space
                    #                     scale_factor=args.patch_size,
                    #                     mode='nearest')[0].cpu().numpy()



                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    from svd import compute_svd_visualize, visualize_qk_attention
                    layerwise_results = {}
                    for layer_id, qkv_out in feat_out_id.items():
                        nb_im, nb_tokens, nhd = qkv_out.shape  # shape: (B, N, 3*D)
                        layer_id = int(layer_id)
                        nh = model.blocks[layer_id].attn.num_heads
                        d = nhd // (3*nh)

                        qkv = qkv_out.reshape(nb_im, nb_tokens, 3, nh, d).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

                        q_cls = q[:, :, 0, :]      # (B, H, D)
                        q_all = q[:, :, 1:, :]     # (B, H, N-1, D)
                        k_cls = k[:, :, 0, :]
                        k_all = k[:, :, 1:, :]

                        # CLS → patch attention logits (B=1 case, so squeeze later)
                        attn3 = torch.matmul(
                            q_cls.unsqueeze(-2),          # (B, H, 1, D)
                            k_all.transpose(-2, -1)       # (B, H, D, N-1)
                        ).squeeze(-2).squeeze(0)          # → (H, N-1)

                        attn3 = attn3 / math.sqrt(q_cls.size(-1))  # scale

                        # Now run your KL reduction function
                        stats = detect_artifact_and_kl_reduction(attn3)

                        layerwise_results[layer_id] = {
                            "artifact_idx": stats["artifact_idx"],
                            "agreement": stats["agreement"],
                            "kl_to_avg": stats["kl_to_avg"],
                            "kl_without_sinks": stats["kl_without_sinks"],
                            "avg_kl_reduction": float(np.mean(stats["kl_reduction"]))
    }
                        layerwise_means = {i: [] for i in range(len(model.blocks))}
                        num_layers = len(model.blocks)
                        mid = num_layers // 2
                        early_kl = []
                        late_kl = []

                        log_path = "kl_reduction_log.txt"

                        with open(log_path, "a") as f:
                            for layer_id, stats in layerwise_results.items():
                                layerwise_means[layer_id].append(stats["avg_kl_reduction"])
                                
                                line = f"Layer {layer_id} Avg KL reduction = {stats['avg_kl_reduction']:.4f}"
                                print(line)
                                f.write(line + "\n")

                                if layer_id < mid:
                                    early_kl.append(stats["avg_kl_reduction"])
                                else:
                                    late_kl.append(stats["avg_kl_reduction"])

                            early_val = np.mean(early_kl) if early_kl else 'N/A'
                            late_val = np.mean(late_kl) if late_kl else 'N/A'

                            early_line = f"Early layers avg KL reduction: {early_val}"
                            late_line = f"Late layers avg KL reduction: {late_val}"

                            print(early_line)
                            print(late_line)

                            f.write(early_line + "\n")
                            f.write(late_line + "\n\n")   # extra newline for readability

                        
                        # flatten (B,H,N,D) → (B*N, D)
                        q = q.transpose(1,2).reshape(nb_im, nb_tokens, -1)
                        k = k.transpose(1,2).reshape(nb_im, nb_tokens, -1)
                        #q = q / q.norm(p=2, dim=-1, keepdim=True)
                        #k = k / k.norm(p=2, dim=-1, keepdim=True)
                        save_dir = os.path.join(vis_folder, "svd", im_name.replace(".jpg", ""))
                        save_path = os.path.join(save_dir, f"layer_{layer_id}.png")
                        visualize_qk_attention(attn3, artifact_idx=stats["artifact_idx"], 
                        save_path=save_path, 
                        title=f"Layer {layer_id} CLS→patch Q×K Attention")
                        # visualize
                        #compute_svd_visualize(q, k, n_components=3, title=f"SVD of Q/K — Layer {layer_id}", save_path=None) 
                    #if args.visualize == "svd":
                    #   compute_svd_visualize(q, k, n_components=2, title="SVD Projection of Q and K")
                    feats1 = None
                    if args.which_features == "k":
                        feats = q[:, 1:, :]  # skip CLS
                        feats1 = k[:, 1:, :]
                        #feats1 = q[:, 1:, :]
                    elif args.which_features == "q":
                        feats = q[:, 1:, :]
                        feats1 = q[:, 1:, :]
                    elif args.which_features == "v":
                        feats = v[:, 1:, :]
                    if 'reg' in args.arch:
                        feats = feats[:, 4:, :]

        # ------------ Apply LOST -------------------------------------------
        if not args.dinoseg:
            # Apply LOST
            # The larger bboxes, the less patches
            # def k_from_bboxes(gt_boxes, img_h, img_w, patch_size,
            #       scale_factor=0.5, k_min=70, k_max=500):
            #     """
            #     Estimate k (number of lowest-degree patches) from GT boxes.
            #     - gt_boxes:  (N,4)  ndarray | tensor  [x0,y0,x1,y1] in pixels
            #     - img_h, img_w:      original image size in pixels
            #     - patch_size:        ViT patch edge length in pixels
            #     Returns an integer between k_min and k_max.
            #     """
            #     if gt_boxes is None or len(gt_boxes) == 0:
            #         return k_min                        # fallback when no GT
            #     # total GT area
            #     areas = (gt_boxes[:,2] - gt_boxes[:,0]) * (gt_boxes[:,3] - gt_boxes[:,1])
            #     gt_area   = areas.sum()
            #     img_area  = img_h * img_w
            #     if gt_area > img_area:
            #         return k_min  # fallback when GT area is larger than image area
            #     # fraction of the image covered by all objects
            #     frac = float(gt_area) / (img_area + 1e-6)

            #     inv_frac = 1.0 - min(max(frac, 0.0), 1.0)

            #     # total number of patches
            #     n_rows = math.ceil(img_h / patch_size)
            #     n_cols = math.ceil(img_w / patch_size)
            #     tot_patches = n_rows * n_cols

            #     # estimate and clamp
            #     k_est = int(inv_frac * scale_factor * tot_patches)
            #     k_clamped = max(k_min, min(k_est, k_max))
            #     return k_clamped

            # The larger bboxes, the more patches
            def k_from_bboxes(gt_boxes, img_h, img_w, patch_size,
                  scale_factor=0.5, k_min=70, k_max=500):
                """
                Estimate k (number of lowest-degree patches) from GT boxes,
                so that larger GT area produces a larger k.

                - gt_boxes:  (N,4) ndarray | tensor [x0,y0,x1,y1] in pixels
                - img_h, img_w: original image size
                - patch_size:   ViT patch edge length
                Returns k between k_min and k_max.
                """
                # fallback when no GT
                if gt_boxes is None or len(gt_boxes) == 0:
                    return k_min

                # total GT area
                areas = (gt_boxes[:,2] - gt_boxes[:,0]) * (gt_boxes[:,3] - gt_boxes[:,1])
                gt_area  = areas.sum()
                img_area = img_h * img_w

                # fraction of the image covered by all objects
                frac = float(gt_area) / (img_area + 1e-6)
                # clamp to [0,1]
                frac = min(max(frac, 0.0), 1.0)

                # total number of patches
                n_rows     = math.ceil(img_h / patch_size)
                n_cols     = math.ceil(img_w / patch_size)
                tot_patches = n_rows * n_cols

                # now larger frac → larger k_est
                k_est = int(frac * scale_factor * tot_patches)

                # clamp into [k_min, k_max]
                k_clamped = max(k_min, min(k_est, k_max))
                return k_clamped
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
                # custom_scores=fused_scores
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
                continue
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
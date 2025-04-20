import torch
import numpy as np

import torch


def detect_artifacts(features: torch.Tensor, method: str = 'threshold', threshold: float = 80.0,
                     gap_threshold: float = 30.0) -> dict:
    """
    Detects artifacts in token features based on L2 norms using the specified method.

    Args:
        features (torch.Tensor): Output features, shape [1, num_tokens, feature_dim].
        method (str): Detection method ('threshold' or 'gap').
        threshold (float): Norm threshold for 'threshold' method (default: 80.0).
        gap_threshold (float): Threshold to flag artifacts in 'gap' method (default: 10.0).

    Returns:
        dict: Contains:
            - artifacts (torch.Tensor): Boolean tensor indicating artifacts, shape [1, num_tokens].
            - indices (list): Indices of artifact tokens.
            - norms (torch.Tensor): L2 norms of tokens, shape [1, num_tokens].
            - top_5_norms (torch.Tensor): Top 5 norms after sorting, shape [5].
            - top_5_indices (list): Indices of top 5 norms.
            - gap (float): Gap between highest norm and fifth highest norm (for 'gap' method).
    """
    # Validate input
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if len(features.shape) != 3 or features.shape[0] != 1:
        raise ValueError("Expected shape [1, num_tokens, feature_dim]")

    # Compute L2 norms
    norms = torch.norm(features, dim=2)  # [1, num_tokens]

    # Sort norms in descending order
    if method == 'threshold':
        # Simple threshold method
        artifacts = norms > threshold
        indices = torch.where(artifacts[0])[0].tolist()

        return {
            "artifacts": artifacts,
            "indices": indices,
            "norms": norms,
        }
    elif method == 'gap':
        sorted_norms, sorted_indices = torch.sort(norms[0], descending=True)

        # Get top 5 norms and indices
        top_5_norms = sorted_norms[:min(5, len(sorted_norms))]  # [5] or fewer
        top_5_indices = sorted_indices[:min(5, len(sorted_norms))].tolist()
        if len(sorted_norms) < 5:
            raise ValueError("Need at least 5 tokens to use 'gap' method.")

        highest_norm = top_5_norms[0]
        fifth_norm = top_5_norms[4]
        gap = highest_norm - fifth_norm

        # Flag artifacts: norms exceeding fifth_norm + gap_threshold
        dynamic_threshold = fifth_norm + gap_threshold
        artifacts = norms > dynamic_threshold
    else:
        raise ValueError("Method must be 'threshold' or 'gap'")

    # Get indices of artifacts
    indices = torch.where(artifacts[0])[0].tolist()

    return {
        "artifacts": artifacts,
        "indices": indices,
        "norms": norms,
        "top_5_norms": top_5_norms,
        "top_5_indices": top_5_indices,
        "gap": gap if method == 'gap' else None
    }



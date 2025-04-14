import torch
import numpy as np


def detect_artifacts(attention, artifact_factor_threshold=10.0):
    """
    Kiểm tra artifacts theo phương pháp relative, dựa trên tensor attention có shape [num_heads, num_tokens].

    Quy trình:
      1. Tính L2 norm cho mỗi patch token, tổng hợp qua các head (norm được tính theo chiều 0).
      2. Tính giá trị trung vị (median) của các norm.
      3. Đánh dấu token là artifact nếu norm của nó lớn hơn
         artifact_factor_threshold * median.
      4. Tính artifact_ratio = số token artifact / tổng số token.
      5. Nếu artifact_ratio > artifact_ratio_threshold (ở đây đặt = 0),
         mẫu được coi là có artifacts (dù chỉ có 1 token).

    Parameters:
      attention (torch.Tensor): Tensor attention có shape [nh, tokens] (ví dụ [16, 1008]).
      artifact_factor_threshold (float): Hệ số nhân với median để xác định token artifact.
                                          Ví dụ, nếu artifact_factor_threshold=10.0, token nào có norm > 10*median sẽ được đánh dấu.
      artifact_ratio_threshold (float): Ngưỡng tỷ lệ token artifact. Ở đây, nếu > 0 thì có artifact.

    Returns:
      artifact_ratio (float): Tỷ lệ token có artifact.
      has_artifacts (bool): True nếu artifact_ratio > artifact_ratio_threshold.
      norms (np.ndarray): Mảng các L2 norm của từng token (shape [tokens]).
      artifact_mask (np.ndarray): Mask Boolean đánh dấu token artifact (shape [tokens]).
    """
    # Tính L2 norm cho từng token, qua các head (dọc theo chiều 0)
    norms = torch.norm(attention, dim=0)  # shape: [tokens]

    min_norm = torch.min(norms).item()
    max_norm = torch.max(norms).item()
    median_norm = torch.median(norms).item()

    print("Min norm:", min_norm)
    print("Max norm:", max_norm)
    print("Median norm:", median_norm)

    # Đánh dấu artifact nếu norm của token > (artifact_factor_threshold * median)
    artifact_mask = norms > (artifact_factor_threshold * median_norm)
    artifact_ratio = artifact_mask.float().mean().item()  # tỷ lệ token artifact
    has_artifacts = artifact_ratio > 0

    return {
        'artifact_ratio': artifact_ratio,
        'has_artifacts': has_artifacts,
        'norms': norms.cpu().numpy(),
        'artifact_mask': artifact_mask.cpu().numpy()
    }

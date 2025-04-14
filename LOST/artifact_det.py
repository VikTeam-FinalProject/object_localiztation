import torch
import numpy as np


def detect_artifacts(tokens, artifact_norm_threshold=150.0, artifact_ratio_threshold=0.02):
    """
    Kiểm tra artifact trong token embeddings đã được truyền vào.

    Parameters:
      tokens (torch.Tensor): Tensor chứa các token embeddings với shape (B, N, D) hoặc (N, D).
                           Chú ý: Các token này phải là các patch token, tức đã loại bỏ token [CLS] nếu có.
      artifact_norm_threshold (float): Ngưỡng L2 norm để coi một token là artifact.
      artifact_ratio_threshold (float): Ngưỡng tỷ lệ token artifact; nếu tỷ lệ artifact vượt qua thì mẫu được coi có artifacts.

    Returns:
      dict: Trả về dictionary bao gồm:
            - 'artifact_ratio': Tỷ lệ phần trăm các token có norm vượt ngưỡng.
            - 'has_artifacts' : Boolean (hoặc tensor Boolean cho batch) cho biết mẫu có chứa artifact hay không.
            - 'norms'         : Các giá trị L2 norm của token, dạng numpy array.
            - 'artifact_mask' : Mask boolean xác định token nào là artifact, dạng numpy array.
    """
    # Tính norm của từng token theo chiều cuối cùng (D)
    if tokens.dim() == 3:
        norms = torch.norm(tokens, dim=-1)  # shape: (B, N)
    elif tokens.dim() == 2:
        norms = torch.norm(tokens, dim=-1)  # shape: (N,)
    else:
        raise ValueError("Tensor tokens phải có shape (N, D) hoặc (B, N, D)")

    artifact_mask = norms > artifact_norm_threshold
    if norms.dim() == 1:
        artifact_ratio = artifact_mask.float().mean().item()
    else:
        artifact_ratio = artifact_mask.float().mean(dim=1)

    if norms.dim() == 1:
        has_artifacts = artifact_ratio > artifact_ratio_threshold
    else:
        has_artifacts = artifact_ratio > artifact_ratio_threshold

    return {
        'artifact_ratio': artifact_ratio,
        'has_artifacts': has_artifacts,
        'norms': norms.cpu().numpy(),
        'artifact_mask': artifact_mask.cpu().numpy()
    }


def check_attention_artifacts(attention, artifact_norm_threshold=150.0, artifact_ratio_threshold=0.02):
    """
    Kiểm tra artifacts trực tiếp từ attention (hay token embeddings) đã được tính sẵn.

    Parameters:
      attention (torch.Tensor): Tensor chứa các patch token embeddings với shape (B, N, D) hoặc (N, D).
      artifact_norm_threshold (float): Ngưỡng L2 norm để coi một token là artifact.
      artifact_ratio_threshold (float): Ngưỡng tỷ lệ token artifact để kết luận mẫu có artifacts hay không.

    Returns:
      dict: Kết quả trả về từ hàm detect_artifacts (bao gồm artifact_ratio, has_artifacts, norms, artifact_mask).
    """
    return detect_artifacts(attention, artifact_norm_threshold, artifact_ratio_threshold)


def check_multiple_attentions_artifacts(attention_items, artifact_norm_threshold=150.0, artifact_ratio_threshold=0.02):
    """
    Kiểm tra artifacts cho một danh sách attention embeddings (các mẫu) theo định dạng thuận tiện.

    Parameters:
      attention_items (list): Danh sách các tuple (im_name, attention_tensor) với attention_tensor có shape (B, N, D) hoặc (N, D).
      artifact_norm_threshold (float): Ngưỡng norm cho artifact.
      artifact_ratio_threshold (float): Ngưỡng tỷ lệ artifact để coi mẫu có artifacts hay không.

    Returns:
      dict: Dictionary kết quả cho từng mẫu, với key là im_name và value là kết quả từ detect_artifacts.
    """
    results = {}
    for im_name, att in attention_items:
        res = check_attention_artifacts(att, artifact_norm_threshold, artifact_ratio_threshold)
        results[im_name] = res
        print(f"Sample '{im_name}': artifact_ratio = {res['artifact_ratio']} -> has_artifacts: {res['has_artifacts']}")
    return results

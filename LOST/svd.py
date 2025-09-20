import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import numpy as np
import os
# Example Q, K (from your Transformer forward pass)
# Assume q, k have already been shaped like [nb_im, nb_tokens, d_model]
# using your reshape code
# q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
# k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)


# def compute_svd_visualize(q, k, n_components=2, title="SVD Projection of Q and K", return_artifact=True):
#     """
#     Perform SVD on concatenated Q and K, plot first two components,
#     and optionally visualize dominant singular vectors (artifacts).
#     """
#     nb_im, nb_tokens, d_model = q.shape
#     print(f"Shapes: Q={q.shape}, K={k.shape}")

#     # flatten across batch+tokens
#     q_flat = q.reshape(-1, d_model).detach().cpu().numpy()
#     k_flat = k.reshape(-1, d_model).detach().cpu().numpy()

#     # concatenate for joint SVD
#     all_feats = np.vstack([q_flat, k_flat])

#     # compute SVD
#     svd = TruncatedSVD(n_components=n_components)
#     proj = svd.fit_transform(all_feats)

#     # split back
#     proj_q = proj[:len(q_flat)]
#     proj_k = proj[len(q_flat):]

#     # plot
#     plt.figure(figsize=(7, 7))
#     plt.scatter(proj_q[:, 0], proj_q[:, 1], alpha=0.3, label="Q", c="blue")
#     plt.scatter(proj_k[:, 0], proj_k[:, 1], alpha=0.3, label="K", c="red")

#     plt.title(title)
#     plt.xlabel("SVD 1")
#     plt.ylabel("SVD 2")
#     plt.legend()

#     if return_artifact:
#         # artifacts are top singular vectors and their explained variance
#         singular_values = svd.singular_values_
#         components = svd.components_

#         # scale components for visualization
#         scale = singular_values / singular_values.max() * 5.0
#         origin = np.zeros(2)

#         for i in range(components.shape[0]):
#             plt.arrow(origin[0], origin[1],
#                       components[i, 0] * scale[i],
#                       components[i, 1] * scale[i],
#                       color="pink", width=1, alpha=0.7,
#                       label=f"Artifact {i+1}")
#         plt.legend()
#         # Print info
#         print("\n=== Artifact Info ===")
#         for i, (sv, ev) in enumerate(zip(singular_values, svd.explained_variance_ratio_)):
#             print(f"Component {i+1}: Singular Value={sv:.4f}, Explained Var={ev:.4f}")

#     plt.show()

#     return svd, proj_q, proj_k


# def compute_svd_visualize(q, k, n_components=2,title="SVD Projection of Q and K",save_path=None):
#     """
#     Perform SVD on concatenated Q and K, and plot first two components.
#     """

#     nb_im, nb_tokens, d_model = q.shape
#     print(f"Shapes: Q={q.shape}, K={k.shape}")

#     # flatten across batch+tokens
#     q_flat = q.reshape(-1, d_model).detach().cpu().numpy()
#     k_flat = k.reshape(-1, d_model).detach().cpu().numpy()

#     # concatenate for joint SVD
#     all_feats = np.vstack([q_flat, k_flat])

#     # compute SVD (PCA-like projection)
#     svd = TruncatedSVD(n_components=n_components)
#     proj = svd.fit_transform(all_feats)

#     # split back
#     proj_q = proj[:len(q_flat)]
#     proj_k = proj[len(q_flat):]

#     # plot
#     plt.figure(figsize=(6, 6))
#     plt.scatter(proj_q[:, 0], proj_q[:, 1], alpha=0.3, label="Q", c="blue")
#     plt.scatter(proj_k[:, 0], proj_k[:, 1], alpha=0.3, label="K", c="red")
#     plt.title(title)
#     plt.xlabel("SVD 1")
#     plt.ylabel("SVD 2")
#     plt.legend()
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=150)
#         plt.close()
#     else:
#         plt.show()


from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import os

# def compute_svd_visualize(q, k, n_components=3, title="3D SVD Projection of Q and K", save_path=None):
#     """
#     Perform SVD on concatenated Q and K, and plot first 2 or 3 components.
#     """

#     nb_im, nb_tokens, d_model = q.shape
#     print(f"Shapes: Q={q.shape}, K={k.shape}")

#     # flatten across batch+tokens
#     q_flat = q.reshape(-1, d_model).detach().cpu().numpy()
#     k_flat = k.reshape(-1, d_model).detach().cpu().numpy()

#     # concatenate for joint SVD
#     all_feats = np.vstack([q_flat, k_flat])

#     # compute SVD
#     svd = TruncatedSVD(n_components=n_components)
#     proj = svd.fit_transform(all_feats)

#     # split back
#     proj_q = proj[:len(q_flat)]
#     proj_k = proj[len(q_flat):]

#     if n_components == 3:
#         # 3D scatter plot
#         fig = plt.figure(figsize=(7, 7))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(proj_q[:, 0], proj_q[:, 1], proj_q[:, 2], alpha=0.3, label="Q", c="blue")
#         ax.scatter(proj_k[:, 0], proj_k[:, 1], proj_k[:, 2], alpha=0.3, label="K", c="red")
#         ax.set_title(title)
#         ax.set_xlabel("SVD 1")
#         ax.set_ylabel("SVD 2")
#         ax.set_zlabel("SVD 3")
#         ax.legend()
#     else:
#         # fallback to 2D scatter
#         plt.figure(figsize=(6, 6))
#         plt.scatter(proj_q[:, 0], proj_q[:, 1], alpha=0.3, label="Q", c="blue")
#         plt.scatter(proj_k[:, 0], proj_k[:, 1], alpha=0.3, label="K", c="red")
#         plt.title(title)
#         plt.xlabel("SVD 1")
#         plt.ylabel("SVD 2")
#         plt.legend()

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=150)
#         print(f"Saved SVD plot to {save_path}")
#         plt.close()
#     else:
#         plt.show()

#     return svd, proj_q, proj_k

import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
import numpy as np

def compute_svd_visualize(q, k, n_components=3, title="3D SVD Projection of Q and K", save_path=None):
    nb_im, nb_tokens, d_model = q.shape
    print(f"Shapes: Q={q.shape}, K={k.shape}")

    # flatten
    q_flat = q.reshape(-1, d_model).detach().cpu().numpy()
    k_flat = k.reshape(-1, d_model).detach().cpu().numpy()

    all_feats = np.vstack([q_flat, k_flat])

    svd = TruncatedSVD(n_components=n_components)
    proj = svd.fit_transform(all_feats)

    proj_q = proj[:len(q_flat)]
    proj_k = proj[len(q_flat):]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=proj_q[:, 0], y=proj_q[:, 1], z=proj_q[:, 2],
        mode='markers', marker=dict(size=3, color='blue'),
        name="Q"
    ))
    fig.add_trace(go.Scatter3d(
        x=proj_k[:, 0], y=proj_k[:, 1], z=proj_k[:, 2],
        mode='markers', marker=dict(size=3, color='red'),
        name="K"
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="SVD 1",
            yaxis_title="SVD 2",
            zaxis_title="SVD 3"
        )
    )

    fig.show()

def compute_svd_single(x, y, n_components=2, title="SVD Projection"):
    """
    Perform SVD on one tensor (Q or K), and plot first two components.
    """
    nb_im, nb_tokens, d_model = x.shape
    print(f"Shape: {x.shape}")

    # flatten
    x_flat = x.reshape(-1, d_model).detach().cpu().numpy()

    # compute SVD
    svd = TruncatedSVD(n_components=n_components)
    proj = svd.fit_transform(x_flat)

    # plot
    plt.figure(figsize=(6, 6))
    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3)
    plt.title(title)
    plt.xlabel("SVD 1")
    plt.ylabel("SVD 2")
    plt.show()

    return svd, proj


# Example usage after you get q and k from the forward pass:
# compute_svd_visualize(q, k)

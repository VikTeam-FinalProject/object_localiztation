from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
from torch import Tensor
from typing import Tuple
from torch.nn import functional as F
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open("outputs/VOC12_train/visualizations/LOST-dinov2_vitl14_pretrain14_k/LOST_2008_000074.jpg_potentials.png")

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
model = AutoModel.from_pretrained('facebook/dinov2-giant')

inputs = processor(images=image, return_tensors="pt")
pixel_vals  = inputs["pixel_values"]        # shape (1, 3, H, W)
_, _, H, W  = pixel_vals.shape
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

inputs = processor(images=image, return_tensors="pt")
# NOTE: the key is "pixel_values" for DINOv2
pixel_values = inputs["pixel_values"]  # shape (1, 3, H, W)

outputs = model(pixel_values, output_attentions=True)  
attentions: Tuple[torch.FloatTensor] = outputs.attentions  
print("attention", attentions)
# attentions is a tuple of length = num_layers; each is (B, heads, L, L)

# 5. Grab the last layer’s attention
last_attn = attentions[-1]  
print("Last layer attention shape:", last_attn.shape)
B, nh, L, _ = last_attn.shape
num_patches = L - 1
patch_size = 14
w_featmap = W // patch_size
h_featmap = H // patch_size
attn = last_attn[0, :, 0, 1:]      # shape (heads, num_patches)

# 4) Optional artifact removal (zero out the most‐common max‐idx across heads)
def det_artifact(attn_heads: torch.Tensor) -> int:
    # attn_heads: (heads, num_patches)
    max_idxs = torch.argmax(attn_heads, dim=1)  # each head’s top patch
    counts  = torch.bincount(max_idxs)
    return int(torch.argmax(counts)) 

attn = attn.reshape(nh,-1)
attn = attn.reshape(nh, w_featmap, h_featmap)  # (heads, 14, 14)

# 6) Upsample back to image resolution via nearest‐neighbor
#    patch_size is 16 for DINOv2-giant

attn_upsampled = (
    F.interpolate(
        attn.unsqueeze(0),  # (1, heads, H, W)
        scale_factor=patch_size,
        mode="nearest",
    )[0]
    .detach()              # <-- detach from the graph
    .cpu()
    .numpy()
)    # (heads, H*patch_size, W*patch_size)

# 7) Quick viz of head 0’s map
import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
plt.imshow(attn_upsampled[0], cmap="inferno")
plt.axis("off")
plt.title("Head 0: CLS→patch attention")
plt.show()     
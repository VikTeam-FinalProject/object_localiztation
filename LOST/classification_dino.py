# from datasets import load_dataset
# from torch.utils.data import Dataset
# import torch
# import numpy as np
# from tqdm import tqdm
# import albumentations as A
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# import torch.nn as nn

# import torch
# import torch.nn as nn
# import os, sys
# current_dir = os.getcwd()

# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# from networks import get_model
# class LinearClassifier(nn.Module):
#     def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
#         super(LinearClassifier, self).__init__()

#         # Các tham số này phải khớp với output của DINOv2
#         self.in_channels = in_channels  # Sẽ là 1024
#         self.width = tokenW             # Sẽ là 32
#         self.height = tokenH            # Sẽ là 32
        
#         # Lớp Conv2d sẽ nhận đầu vào có số kênh bằng embedding_dim
#         self.classifier = nn.Conv2d(in_channels, num_labels, kernel_size=(1, 1))

#     def forward(self, embeddings):
#         B, N, D = embeddings.shape
  
#         assert D == self.in_channels, f"Embedding dimension ({D}) does not match in_channels ({self.in_channels})"

#         assert N == self.height * self.width, f"Number of patches ({N}) does not match tokenH*tokenW ({self.height*self.width})"

#         embeddings = embeddings.reshape(B, self.height, self.width, D)

#         embeddings = embeddings.permute(0, 3, 1, 2)

#         return self.classifier(embeddings)


# class Dinov2WithClassifiers(nn.Module):
#     """
#     A wrapper class that combines a DINOv2 model with a linear classifier head.
#     This is typically used for dense prediction tasks like semantic segmentation.
#     """
#     def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1, device='cpu'):
#         super(Dinov2WithClassifiers, self).__init__()
#         self.dinov2 = get_model('dinov2_vitl14_pretrain', patch_size=14, device=device)
#         self.linear_cls = LinearClassifier(in_channels, tokenW, tokenH, num_labels)
#         self.device = device
#         # It's good practice to define the loss function once in the constructor.
#         self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        
#     def forward(self, pixel_values, labels=None):
#         """
#         Forward pass of the model.

#         Args:
#             pixel_values (torch.Tensor): Input images of shape (N, C, H, W).
#             labels (torch.Tensor, optional): Ground truth labels of shape (N, H, W) or (N, 1, H, W).
#                                              Defaults to None.

#         Returns:
#             If labels are provided (i.e., during training):
#                 A tuple containing:
#                 - loss (torch.Tensor): The calculated cross-entropy loss.
#                 - logits (torch.Tensor): The output logits from the classifier.
#             If labels are not provided (i.e., during inference):
#                 - logits (torch.Tensor): The output logits from the classifier.
#         """

#         # Get patch embeddings from the DINOv2 model.
#         # The `forward_features` method returns a dictionary of tensors.
#         # We are interested in 'x_norm_patchtokens' for dense prediction.
#         with torch.no_grad(): 
#             outputs = self.dinov2.forward_features(pixel_values)
        
#         features = outputs['x_norm_patchtokens']
#         logits = self.linear_cls(features)

#         # Calculate loss if labels are provided.
#         if labels is not None:
#             labels = labels.to(self.device)
            
#             # The target for CrossEntropyLoss should be a LongTensor and have shape (N, H, W).
#             # If labels have a channel dimension, we squeeze it.
#             if labels.dim() == 4 and labels.shape[1] == 1:
#                 labels = labels.squeeze(1)
#             logits_for_loss = nn.functional.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)

#             loss = self.loss_fct(logits_for_loss, labels.long())
#             return loss, logits
        

#         return logits



# class SegmentationDataset(Dataset):
#   def __init__(self, dataset, transform):
#     self.dataset = dataset
#     self.transform = transform

#   def __len__(self):
#     return len(self.dataset)

#   def __getitem__(self, idx):
#     item = self.dataset[idx]
#     original_image = np.array(item["image"])
#     original_segmentation_map = np.array(item["label"])

#     transformed = self.transform(image=original_image, mask=original_segmentation_map)
#     image, target = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])

#     # convert to C, H, W
#     image = image.permute(2,0,1)

#     # Return only the tensors needed for training
#     return image, target
  
# dataset = load_dataset("EduardoPacheco/FoodSeg103")

# train_set = dataset['train']
# test_set = dataset['validation']
# learning_rate = 5e-5
# epochs = 10
# patch_size = 14
# pbar = tqdm(train_set)
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print("Using device:", device)

# model = Dinov2WithClassifiers(in_channels=1024, num_labels = 104,device=device)
# optimizer = AdamW(model.parameters(), lr=learning_rate)
# with open("experiments/class.txt", "r", encoding="utf-8") as f:
#     class_names = [line.strip() for line in f]

# #===================================


# ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
# ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# train_transform = A.Compose([

#     A.Resize(width=448, height=448),
#     A.HorizontalFlip(p=0.5),
#     A.Normalize(mean=ADE_MEAN, std=ADE_STD),
# ])

# val_transform = A.Compose([
#     A.Resize(width=448, height=448),
#     A.Normalize(mean=ADE_MEAN, std=ADE_STD),

# ])

# #===================================


# train_dataset = SegmentationDataset(dataset["train"], transform=train_transform)
# val_dataset = SegmentationDataset(dataset["validation"], transform=val_transform)


# def collate_fn(batch):
#     # Simplified collate_fn
#     pixel_values = torch.stack([item[0] for item in batch], dim=0)
#     labels = torch.stack([item[1] for item in batch], dim=0)
#     return {"pixel_values": pixel_values, "labels": labels}

# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
# val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
# batch = next(iter(train_dataloader))
# # for k,v in batch.items():
# #   if isinstance(v,torch.Tensor):
# #     print(k,v.shape)

# #===========================================

# def process_batch(model, batch, device):
#     """
#     Chạy forward pass trên MỘT batch dữ liệu và trả về loss tensor.
    
#     Hàm này được tối ưu cho việc gọi bên trong vòng lặp training.
#     """
#     # 1. Chuyển dữ liệu của batch sang device
#     pixel_values = batch["pixel_values"].to(device)
#     labels = batch["labels"].to(device)

#     # 2. Forward pass để lấy loss và logits
#     # Model đã ở chế độ train() và gradient đã được bật ở vòng lặp ngoài
#     loss, logits = model(pixel_values=pixel_values, labels=labels)
    
#     return loss, logits

# def train_loop(model, dataloader, optimizer, device, num_epochs):
#     """
#     Vòng lặp huấn luyện chính, gọi hàm helper để xử lý từng batch.
#     """
#     model.to(device)

#     for epoch in range(num_epochs):
#         print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        
#         model.train() 
#         running_loss = 0.0

#         for batch in tqdm(dataloader):
#             optimizer.zero_grad()
#             loss, _ = process_batch(model, batch, device)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             # torch.cuda.empty_cache()
#             # print(loss.item()) # More efficient than printing the whole tensor

#         # In loss trung bình của epoch
#         epoch_loss = running_loss / len(dataloader)
#         print(f"Epoch Loss: {epoch_loss:.4f}\n")

#     print("Hoàn tất huấn luyện!")

# train_loop(
#     model=model, 
#     dataloader=train_dataloader, 
#     optimizer=optimizer, 
#     device=device, 
#     num_epochs=10
# )

from datasets import load_dataset
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import torch
import os, sys
current_dir = os.getcwd()

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from networks import get_model
class AttentionLinearClassifier(nn.Module):
    """
    A lightweight classification head for ViT-style blocks.
    It consumes Q, K, V from the *last* Transformer layer.

    Expected shapes
    ---------------
    q, k, v : (B, H, T, d_h)
        B  – batch size
        H  – number of heads
        T  – #tokens (CLS + patches + optional regs)
        d_h – head dimension (d_model // H)
    """

    def __init__(
        self,
        num_tokens: int,             # T  (CLS + patches …)
        head_dim: int,               # d_h
        num_heads: int,              # H
        num_labels: int = 1,         # 1 = binary, >1 = multi-class
        device: str = "cuda",
        agg: str = "mean",       # 'cls_mean' | 'global_avg' | 'flatten'
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.head_dim   = head_dim
        self.num_heads  = num_heads
        self.agg        = agg

        if agg == "flatten":
            in_feats = num_heads * num_tokens * head_dim
        else:                           
            in_feats = head_dim
        self.classifier = nn.Linear(in_feats, num_labels, device=device)
        self.classifier = self.classifier.to(device)


    def forward(
        self,
        q_cls: torch.Tensor,      # (B, H, d_h)
        k: torch.Tensor,          # (B, H, N, d_h)
        v: torch.Tensor           # (B, H, N, d_h)
    ) -> torch.Tensor:

        B, H, d_h = q_cls.shape
        _, _, N, _ = k.shape

        # sanity
        assert H == self.num_heads and d_h == self.head_dim

        # 1) attention weights α_{b,h,n} = softmax(Q_cls·K^T / √d_h)
        scale = 1.0 / math.sqrt(d_h)
        attn  = torch.einsum("bhd, bhnd -> bhn", q_cls, k) * scale   # (B,H,N)
        attn  = nn.functional.softmax(attn, dim=-1)

        ctx = torch.einsum("bhn, bhnd -> bhd", attn, v)              # (B,H,d_h)
        # 3) aggregate heads
        if self.agg == "mean":
            vec = ctx.mean(dim=1)                                    # (B,d_h)
        else:  # 'flatten'
            vec = ctx.reshape(B, H * d_h)                            # (B,H·d_h)
        # print(self.classifier.weight.device)
        return self.classifier(vec)


def make_qkv_hook(storage: dict, num_heads: int):
    """
    Builds a forward-hook that:
      • takes the raw (B, T, 3·C) tensor from the qkv linear layer,
      • reshapes it to (3, B, H, T, C / H),
      • stores q, k, v separately in `storage`.
    """
    def _hook(module, inp, out):
        # out  : (B, T, 3·C)
        B, T, _3C = out.shape
        head_dim  = _3C // 3 // num_heads      # C/H

        qkv = (
            out               # (B, T, 3·C)
            .reshape(B, T, 3, num_heads, head_dim)
            .permute(2, 0, 3, 1, 4)            # (3, B, H, T, C/H)
        )
        storage["q"], storage["k"], storage["v"] = qkv.unbind(dim=0)
        # If you only need qkv later and want to avoid autograd overhead:
        for k in ("q", "k", "v"):
            storage[k] = storage[k].detach()
    return _hook

class Dinov2WithClassifiers(nn.Module):
    """
    A wrapper class that combines a DINOv2 model with a linear classifier head.
    This is typically used for dense prediction tasks like semantic segmentation.
    """
    def __init__(self, num_labels=1, device='cpu'):
        super(Dinov2WithClassifiers, self).__init__()
        self.dinov2 = get_model('dinov2_vitl14_pretrain', patch_size=14, device=device)
        print(self.dinov2.patch_embed.num_patches)
        
        self.linear_cls = AttentionLinearClassifier(
            num_tokens=self.dinov2.patch_embed.num_patches +1 ,
            head_dim=self.dinov2.embed_dim   // self.dinov2.num_heads,
            num_heads=self.dinov2.num_heads,
            num_labels=num_labels,
            device=device
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        
    def forward(self, pixel_values, labels=None):
        """
        Forward pass of the model.

        Args:
            pixel_values (torch.Tensor): Input images of shape (N, C, H, W).
            labels (torch.Tensor, optional): Ground truth labels of shape (N, H, W) or (N, 1, H, W).
                                             Defaults to None.

        Returns:
            If labels are provided (i.e., during training):
                A tuple containing:
                - loss (torch.Tensor): The calculated cross-entropy loss.
                - logits (torch.Tensor): The output logits from the classifier.
            If labels are not provided (i.e., during inference):
                - logits (torch.Tensor): The output logits from the classifier.
        """

        # Get patch embeddings from the DINOv2 model.
        # The `forward_features` method returns a dictionary of tensors.
        # We are interested in 'x_norm_patchtokens' for dense prediction.
        feat_out = {}
        last_attn   = self.dinov2.blocks[-1].attn           # convenience handle
        num_heads   = last_attn.num_heads
        hook_handle = last_attn.qkv.register_forward_hook(
            make_qkv_hook(feat_out, num_heads)
        )
        _ = self.dinov2.get_last_selfattention(pixel_values)
        hook_handle.remove()
        q_cls = feat_out.get("q")[:, :, 0, :]
        q_no_cls = feat_out.get("q")[:, :, 1:, :]
        v_all = feat_out.get("v")[:, :, 1:, :]
        logits = self.linear_cls(q_cls,q_no_cls,v_all)

        # Calculate loss if labels are provided.
        if labels is not None:
            if labels.dim() > 1:
                labels = labels.squeeze()  # Squeeze mọi chiều thừa
            
            assert labels.dim() == 1, f"Expected labels to have shape (N,), but got {labels.shape}"

            loss = self.loss_fct(logits, labels.long())
            return loss, logits
        

        return logits

class ClassificationDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset   = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img  = np.array(item["image"])
        label = torch.tensor(item["label"], dtype=torch.long)

        # chỉ transform trên image, không có mask
        augmented = self.transform(image=img)
        img_tensor= augmented["image"]  # đã là (C,H,W) và FloatTensor
    
        return img_tensor, label
    
dataset = load_dataset("flwrlabs/caltech101")
split_ds = dataset["train"].train_test_split(test_size=0.2, seed=42)

train_set = split_ds['train']
val_set = split_ds['test']
learning_rate = 5e-5
epochs = 10
patch_size = 14
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)
model = Dinov2WithClassifiers(num_labels = 256,device=device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
with open("experiments/class.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

#===================================


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

def ensure_three_channels(img, **kwargs):
    # nếu grayscale (H×W) hoặc (H×W×1), lặp thành 3 channel
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    return img

train_transform = A.Compose([
    A.Lambda(image=ensure_three_channels),      
    A.Resize(width=518, height=518),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ToTensorV2(),            # ← convert H×W×C numpy → C×H×W tensor
])

val_transform = A.Compose([
    A.Resize(width=518, height=518),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ToTensorV2(),            # ← convert H×W×C numpy → C×H×W tensor
])

#===================================


train_dataset = ClassificationDataset(train_set, transform=train_transform)
val_dataset = ClassificationDataset(val_set, transform=val_transform)


def collate_fn(batch):
    # Simplified collate_fn
    pixel_values = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    return {"pixel_values": pixel_values, "labels": labels}

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
batch = next(iter(train_dataloader))
# for k,v in batch.items():
#   if isinstance(v,torch.Tensor):
#     print(k,v.shape)

#===========================================

def process_batch(model, batch, device):
    """
    Chạy forward pass trên MỘT batch dữ liệu và trả về loss tensor.
    
    Hàm này được tối ưu cho việc gọi bên trong vòng lặp training.
    """
    # 1. Chuyển dữ liệu của batch sang device
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    # 2. Forward pass để lấy loss và logits
    # Model đã ở chế độ train() và gradient đã được bật ở vòng lặp ngoài
    loss, logits = model(pixel_values=pixel_values, labels=labels)
    
    return loss, logits

def train_loop(model, dataloader, optimizer, device, num_epochs):
    """
    Vòng lặp huấn luyện chính, gọi hàm helper để xử lý từng batch.
    """
    model.to(device)

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        
        model.train() 
        running_loss = 0.0

        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            loss, _ = process_batch(model, batch, device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # torch.cuda.empty_cache()
            # print(loss.item()) # More efficient than printing the whole tensor

        # In loss trung bình của epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch Loss: {epoch_loss:.4f}\n")

    print("Hoàn tất huấn luyện!")

train_loop(
    model=model, 
    dataloader=train_dataloader, 
    optimizer=optimizer, 
    device=device, 
    num_epochs=10
)


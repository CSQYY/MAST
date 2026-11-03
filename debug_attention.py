# debug_attention.py - 添加详细形状检查
import torch
import sys
sys.path.insert(0, r'D:/code/2026/MAST-main')

from attention_visualizer import MASTAttentionVisualizer
from dataloader_v5 import DataLoader
import torch.utils.data as data

VIDEO_ROOT = 'E:/dataset/ZDL/DOLOS/image/'
VIDEO_NAME = 'AN_WILTY_EP15_lie12'
CKPT_PATH = r'C:/Users/10244/Desktop/vit6/6AU_bestweight35.pth'
DEVICE = torch.device('cuda:0')

# 加载模型
print("Loading model...")
model = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.eval().to(DEVICE)

# 初始化
vis = MASTAttentionVisualizer(model, device=DEVICE)

# 加载数据
test_dataset = DataLoader(
    video_folder=VIDEO_ROOT,
    img_size=224,
    frame_len=64,
    video_names=[VIDEO_NAME]
)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
sample = next(iter(test_loader))
img, au, limb, label, _ = sample

# 扩展 batch
video_input = img.repeat(2, 1, 1, 1, 1).to(DEVICE)
au_feat = au.repeat(2, 1, 1, 1).to(DEVICE)
limb_feat = limb.repeat(2, 1, 1, 1).to(DEVICE)

print(f"\nInput shapes:")
print(f"  video: {list(video_input.shape)}")
print(f"  au: {list(au_feat.shape)}")
print(f"  limb: {list(limb_feat.shape)}")

# 测试提取
print("\n=== Testing AU attention ===")
attn_au = vis.extract_attention(video_input, au_feat, limb_feat, modality='au')
if attn_au is not None:
    print(f"✓ AU attention shape: {attn_au.shape}")
    print(f"  Min: {attn_au.min():.4f}, Max: {attn_au.max():.4f}, Mean: {attn_au.mean():.4f}")
    if attn_au.shape[0] == 196:
        print(f"  ✓ Correct patch count!")
    else:
        print(f"  ✗ Wrong patch count! Expected 196, got {attn_au.shape[0]}")
else:
    print("✗ AU attention failed")

print("\n=== Testing Limb attention ===")
attn_limb = vis.extract_attention(video_input, au_feat, limb_feat, modality='limb')
if attn_limb is not None:
    print(f"✓ Limb attention shape: {attn_limb.shape}")
    print(f"  Min: {attn_limb.min():.4f}, Max: {attn_limb.max():.4f}, Mean: {attn_limb.mean():.4f}")
    if attn_limb.shape[0] == 196:
        print(f"  ✓ Correct patch count!")
    else:
        print(f"  ✗ Wrong patch count! Expected 196, got {attn_limb.shape[0]}")
else:
    print("✗ Limb attention failed")

vis.remove_hooks()
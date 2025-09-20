from transformers import AutoImageProcessor, AutoModel, AutoConfig
from PIL import Image
import requests
from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTAttention
import cv2
import numpy as np
from preprocessor import CustomImageProcessor  
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

device = torch.device("cpu")
# url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/STS-124_launch_from_a_distance.jpg/800px-STS-124_launch_from_a_distance.jpg'
image = Image.open("/home/qyo9735/DinoV3/transformers/3.jpg")

name="3.png"
raw_image = image.convert("RGB").resize((1024, 1024))  # Ensure image is in RGB format
raw_image.save(f"vq_pca/input/{name}")

print(raw_image.size)  # Print original image size

config = AutoConfig.from_pretrained('facebook/dinov3-vit7b16-pretrain-lvd1689m')
# config._attn_implementation="sdpa"
# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-giant', cache_dir="/data2/onkar/llava")
model = AutoModel.from_pretrained('facebook/dinov3-vit7b16-pretrain-lvd1689m', config=config, cache_dir="/data2/onkar/llava")
model.to(device)

processor = CustomImageProcessor(1024, device = device)



inputs = processor(images=image, return_tensors="pt")
out = model(**inputs, output_attentions=False, output_hidden_states=True)

for i in range(1,5):

# final_out = final_out[0,5:]

    outputs = out.hidden_states[-i]

    final_out = outputs[0, 5:, :].detach().cpu().numpy()  # shape = [num_patches, hidden_dim]


    pca = PCA(n_components=3)
    final_out_pca = pca.fit_transform(final_out)  # shape = [num_patches, 3]



    # Reshape back into 2D grid
    h = int(np.sqrt(final_out.shape[0]))  # num_patches must be a square
    w = h
    final_out_pca = final_out_pca.reshape(h, w, 3)

    # Normalize to 0–1 for visualization
    final_out_pca = (final_out_pca - final_out_pca.min()) / (final_out_pca.max() - final_out_pca.min())
    # plt.imsave("./small_image.png", final_out_pca)

    final_out_pca = torch.from_numpy(final_out_pca).permute(2, 0, 1).unsqueeze(0)  
    # -> shape [1, 3, h, w] for interpolate

    # Get original image size
    orig_h, orig_w = raw_image.size[1], raw_image.size[0]  # PIL gives (W, H)

    # Upsample to original size
    final_out_pca_resized = F.interpolate(
        final_out_pca,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False
    )

    # Convert back to [H, W, 3] for visualization
    final_out_pca_resized = final_out_pca_resized.squeeze(0).permute(1, 2, 0).numpy()

    if i == 1:
        j=4
    if i==2:
        j=3
    if i==3:
        j=2
    if i==4:
        j=1
    plt.imsave(f'/home/qyo9735/DinoV3/transformers/vq_pca/vq{j}/{name}', final_out_pca_resized)

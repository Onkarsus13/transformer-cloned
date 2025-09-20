from transformers import AutoImageProcessor, AutoModel, AutoConfig
from PIL import Image
import requests
from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTAttention
import cv2
import numpy as np
from preprocessor import CustomImageProcessor  

# url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/STS-124_launch_from_a_distance.jpg/800px-STS-124_launch_from_a_distance.jpg'
image = Image.open("/data2/onkar/transformers/image.png")

raw_image = image.convert("RGB")  # Ensure image is in RGB format

print(raw_image.size)  # Print original image size

config = AutoConfig.from_pretrained('facebook/dinov3-vit7b16-pretrain-lvd1689m')
config.attn_implementation="sdpa"
# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-giant', cache_dir="/data2/onkar/llava")
model = AutoModel.from_pretrained('facebook/dinov3-vit7b16-pretrain-lvd1689m', config=config, cache_dir="/data2/onkar/llava")

# print(model)

processor = CustomImageProcessor(512)
# print(model)

cross_attn_maps = []

# Define a forward hook for cross-attention layers
def cross_attn_hook(module, input, output):
    # print(output[1])
    cross_attn_maps.append(output[0].detach().cpu())

# print(model)
# Find and register hooks on all cross-attention layers
hooks = []
for name, module in model.named_modules():
    if "attention" in name and isinstance(module, DINOv3ViTAttention):
        # print(module)
        hooks.append(module.register_forward_hook(cross_attn_hook))


inputs = processor(images=image, return_tensors="pt").to('cuda:1')

outputs = model(**inputs, output_attentions=True, output_hidden_states=True).hidden_states[-1]


dog_image = Image.open('man_dog.jpg').convert("RGB")
dog_inputs = processor(images=dog_image, return_tensors='pt')
dog_outputs = model(**dog_inputs, output_attentions=True, output_hidden_states=True).hidden_states[-1]

final_out = dog_outputs[0]@outputs[0].T

final_out = final_out[0,5:]

# print(final_out.shape)


print(len(cross_attn_maps))
# Pick the last cross-attention map (from last layer and token)
if len(cross_attn_maps) > 0:
    # last_attn = cross_attn_maps[-1]  # shape: [1, num_heads, tgt_len, src_len]'
    # # print(len(cross_attn_maps), "cross-attention maps captured.")
    # print(last_attn.shape)
    # # Take the mean over heads and pick the last generated token (could adjust as needed)
    # attn_map = last_attn[0].sum(0)[0].numpy()  # shape: [src_len] ffmpeg ffmeg
    # print(attn_map.shape)

    # img_attn = attn_map[5:]
    img_attn = outputs[0,5:,0].detach().numpy()
    img_attn = final_out.detach().numpy()
    grid_size = int(np.sqrt(img_attn.shape[0]))
    attn_grid = img_attn.reshape(grid_size, grid_size)
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)  # normalize
    attn_uint8 = np.uint8(attn_grid * 255)
    attn_color = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
    attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)
    attn_color = cv2.resize(attn_color, (grid_size*grid_size, grid_size*grid_size), interpolation=cv2.INTER_CUBIC)

    # Show attention overlay on image
    img_vis = raw_image.resize((grid_size*grid_size, grid_size*grid_size))
    img_np = np.array(img_vis)
    overlay = cv2.addWeighted(img_np, 0.8, attn_color, 0.4, 0)
    overlay_img = Image.fromarray(overlay)                    
    overlay_img.save("attention_overlay_projected.png")



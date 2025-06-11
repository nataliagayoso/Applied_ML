from PIL import Image
import numpy as np
import io
import torch

def preprocess_image(file_bytes, for_cnn=False):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)

    if for_cnn:
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 224, 224]
        return tensor
    else:
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.flatten().reshape(1, -1)

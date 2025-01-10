import numpy as np
from transformers import BeitFeatureExtractor, BeitModel
from PIL import Image
import torch
import clip

# BEiT
model_beit_name = "microsoft/beit-base-patch16-224-pt22k"
extractor_beit = BeitFeatureExtractor.from_pretrained(model_beit_name)
model_beit = BeitModel.from_pretrained(model_beit_name)

# Trích xuất đặc trưng theo từng batch ảnh
def extract_beit_feature(image_paths, general=False):
  images = [Image.open(path).convert('RGB') for path in image_paths]
  # Chuyển đổi ảnh thành các đầu vào phù hợp với BEiT
  inputs = extractor_beit(images=images, return_tensors="pt")

  # Lấy đặc trưng từ mô hình BEiT
  with torch.no_grad():
      outputs = model_beit(**inputs)

  if general: # sử dụng patch 0
    features = outputs.last_hidden_state[:,0,:]  # Tensor có shape (batch_size, hidden_size)
  else:
    features = outputs.last_hidden_state.mean(dim=1)  # Tensor có shape (batch_size, hidden_size)
  return features

#Tạo model xử lý ảnh clip
class ImageEmbedding():
    def __init__(self, backbone="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(backbone, device=self.device)
        # self.model, self.preprocess = clip.load('RN101', device=self.device)

    def __call__(self, image):
        # image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_feature = self.model.encode_image(image_input)[0]

        return image_feature.detach().cpu().numpy()

clip_extractor_rn = ImageEmbedding(backbone="RN101")
clip_extractor_vit = ImageEmbedding()

def extract_clip_feature(image_paths, general=False):
  images = [Image.open(path).convert('RGB') for path in image_paths]
  features = []
  for image in images:
    if general:
      feature = clip_extractor_rn(image)
    else:
      feature = clip_extractor_vit(image)

    features.append(feature)
  features = np.array(features)
  return features


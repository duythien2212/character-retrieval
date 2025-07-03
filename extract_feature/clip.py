#Tạo model xử lý ảnh clip
import clip
import numpy as np
import torch

from extract_feature.ai_model import AIModel
from PIL import Image


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
    

class CLIP(AIModel):
  def __init__(self):
    self.model_rn = ImageEmbedding(backbone="RN101")
    self.model_vit = ImageEmbedding()
    self.name = "CLIP"
    self.name_general = "CLIP_rn"
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

  def __call__(self, image_paths, general=False):
    images = [Image.open(path).convert('RGB') for path in image_paths]
    features = []
    for image in images:
      if general:
        feature = self.model_rn(image)
      else:
        feature = self.model_vit(image)

      features.append(feature)
    features = np.array(features)
    return features
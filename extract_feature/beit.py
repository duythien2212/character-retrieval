from transformers import BeitFeatureExtractor, BeitModel
from extract_feature.ai_model import AIModel
import torch
from PIL import Image



class BEiT(AIModel):
  def __init__(self):
    self.model_name = "microsoft/beit-base-patch16-224-pt22k"
    self.extractor = BeitFeatureExtractor.from_pretrained(self.model_name)
    self.model = BeitModel.from_pretrained(self.model_name)
    self.name = "BEiT"
    self.name_general = "BEiT_s"
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

  def __call__(self, image_paths, general=False):
    images = [Image.open(path).convert('RGB') for path in image_paths]
    inputs = self.extractor(images=images, return_tensors="pt")

    # Lấy đặc trưng từ mô hình BEiT
    with torch.no_grad():
        outputs = self.model(**inputs)

    if general: # sử dụng patch 0
      features = outputs.last_hidden_state[:,0,:]  # Tensor có shape (batch_size, hidden_size)
    else:
      features = outputs.last_hidden_state.mean(dim=1)  # Tensor có shape (batch_size, hidden_size)
    return features.numpy()
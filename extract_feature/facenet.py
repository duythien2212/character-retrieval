import numpy as np
from extract_feature.ai_model import AIModel
from deepface import DeepFace


class FaceNet(AIModel):
  def __init__(self):
    self.model = DeepFace.build_model(model_name="Facenet")
    self.name = "Facenet"

  def __call__(self, image_paths, general=False):
    features = []
    for i, image_path in enumerate(image_paths):
      try:
        embedding = DeepFace.represent(image_path, model_name="Facenet", detector_backend='mtcnn')[0]['embedding']
        features.append(embedding)
      except Exception as e:
        print(f"Query {i+1} can't be detected face! Error: {e}")
        features.append(np.zeros(self.model.output_shape[-1])) # Append zeros if face not detected

    features = np.array(features)
    return features
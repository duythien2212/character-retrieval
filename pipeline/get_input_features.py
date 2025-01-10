from PIL import Image
from deepface import DeepFace
import numpy as np
from extract_feature.without_face import extract_clip_feature, extract_beit_feature

def get_input_features(list_query, feature_with_face, feature_without_face, general):
  input_features_with_face = []
  cant_detect_face_paths = []
  for i, image_path in enumerate(list_query):
    try:
      # img = Image.open(image_path)
      # img = np.asarray(img)
      image = Image.open(image_path).convert('RGB')
      faces_features = DeepFace.represent(np.array(image), model_name= feature_with_face, detector_backend='mtcnn',
                                          align = False, enforce_detection=True)
      # faces_features = DeepFace.represent(image_path, model_name= feature_with_face, detector_backend='mtcnn')
      if len(faces_features) > 1:
        print("Face not clear")
      for face_features in faces_features:
        input_features_with_face.append(face_features['embedding'])

    except Exception as e:
      print(f"Query {i+1} can't be detected face!")
      cant_detect_face_paths.append(image_path)

    if cant_detect_face_paths:
      if feature_without_face == 'CLIP':
        input_features_without_face = extract_clip_feature(cant_detect_face_paths, general=general)
      elif feature_without_face == 'BEiT':
        input_features_without_face = extract_beit_feature(cant_detect_face_paths, general=general)
      else:
        print("Feature not found")
    else:
      input_features_without_face = []

  # if len(input_features_without_face) > 0:
  #   if feature_without_face == 'CLIP':
  #     input_features_without_face = np.concatenate(input_features_without_face, axis=0)
  #   elif feature_without_face == 'BEiT':
  #     input_features_without_face = torch.cat(input_features_without_face, dim=0).numpy()

  return input_features_with_face, input_features_without_face
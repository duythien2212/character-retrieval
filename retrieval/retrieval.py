import os
import pickle

import numpy as np

from extract_feature import normalize_data


class RetrievalModel():
  def __init__(self, index_path, face_model, non_face_model, merge_model, query_expansion):
    self.face_model = face_model if face_model is not None else None
    self.non_face_model = non_face_model if non_face_model is not None else None
    self.merge_model = merge_model
    self.query_expansion = query_expansion
    self.face_database = None
    self.non_face_database = None


  def load_database(self, index_path, general = None, remove_type = None):
    # Load pre-computed features and FAISS indexes
    if self.face_model is not None:
      face_features_name = self.face_model.name
      if remove_type:
        face_features_name = f'{self.face_model.name}_{remove_type}'
      face_features_path = os.path.join(index_path, 'With_Face', f'{face_features_name}.pkl')
      self.face_database = self.load_database_from_path(face_features_path)

    if self.non_face_model is not None:
      non_face_features_name = self.non_face_model.name if not general else self.non_face_model.name_general
      if remove_type:
        non_face_features_name = f'{self.non_face_model.name}_{remove_type}'
      non_face_features_path = os.path.join(index_path, 'Without_Face_All', f'{non_face_features_name}.pkl')
      self.non_face_database = self.load_database_from_path(non_face_features_path)


  def load_database_from_path(self, path):
    try:
      with open(path, 'rb') as f:
        database = pickle.load(f)
    except FileNotFoundError:
      print("Feature or index files not found. Please run feature extraction and indexing first.")
      return None
    return database


  def get_embeddings(self, shot_names, database):
    result = []
    if database is not None and database.get('map_index') is not None:
      db_shot_names = database['map_index']
      db_shot_names = [shot_name.split('.')[0] for shot_name in db_shot_names]

      name_to_index = dict()
      for i, name in enumerate(db_shot_names):
        name_to_index[name] = i

      for shot_name in shot_names:
        if shot_name in name_to_index:
          index = name_to_index[shot_name]
          embedding = database['index'].reconstruct(index)
          result.append(embedding)
    return result

  def get_face_embeddings(self, shot_names):
    return self.get_embeddings(shot_names, self.face_database)


  def get_non_face_embeddings(self, shot_names):
    return self.get_embeddings(shot_names, self.non_face_database)


  def extract_face_feature(self, image_paths):
    # Extract face features from a given image path
    face_features = []
    for image_path in image_paths:
      try:
        embedding = self.face_model([image_path])
        if len(embedding) > 1:
          print(f"Face not clear in {image_path}")
        face_features.append(embedding[0])
      except Exception as e:
        print(f"Face not detected in {image_path}: {e}")
        pass

    return np.array(face_features)


  def extract_non_face_feature(self, image_paths):
    # Extract non-face features from a given image path
    non_face_features = []
    for image_path in image_paths:
      try:
        embedding = self.non_face_model([image_path])
        non_face_features.append(embedding[0]) # Append the embedding itself, not a list containing it
      except Exception as e:
        print(f"Non-face not detected in {image_path}: {e}")
        pass # Or: features.append(np.zeros(dimension_of_non_face_model))

    return np.array(non_face_features)


  def retrieval(self, image_paths, k=1000, use_query_expansion=True):
    face_features = []
    non_face_features = []

    if self.face_model is not None and self.face_database.get('index') is not None:
      face_features = self.extract_face_feature(image_paths)

    if self.non_face_model is not None and self.non_face_database.get('index') is not None:
      non_face_features = self.extract_non_face_feature(image_paths)

    face_features = np.array(face_features)
    non_face_features = np.array(non_face_features)

    result = self.retrieval_with_embedding(face_features, non_face_features, k, use_query_expansion)
    return result


  def retrieval_with_embedding(self, face_embeddings, non_face_embeddings, k=1000, use_query_expansion=True):
    results_with_face = []
    D_with_face = []
    results_without_face = []
    D_without_face = []

    if self.face_database is not None and self.face_database.get('index') is not None and face_embeddings is not None and face_embeddings.shape[0] > 0:
      D_with_face, I_with_face = self.face_database['index'].search(normalize_data(face_embeddings), k)
      for query_index in I_with_face:
        results_with_face.append([self.face_database['map_index'][i] for i in query_index])

    if self.non_face_database is not None and self.non_face_database.get('index') is not None and non_face_embeddings is not None and non_face_embeddings.shape[0] > 0:
      D_without_face, I_without_face = self.non_face_database['index'].search(normalize_data(non_face_embeddings), k)
      for query_index in I_without_face:
        results_without_face.append([self.non_face_database['map_index'][i] for i in query_index])

    if self.merge_model is not None:
      results = self.merge_model.merge(results_with_face, results_without_face, D_with_face, D_without_face)
      results = [result.split('.')[0] for result in results]
      if self.query_expansion is not None and use_query_expansion == True:
        face_retrieved_embeddings = self.get_face_embeddings(results)
        non_face_retrieved_embeddings = self.get_non_face_embeddings(results)

        new_face_queries = self.query_expansion(face_embeddings, face_retrieved_embeddings)
        new_non_face_queries = self.query_expansion(non_face_embeddings, non_face_retrieved_embeddings)

        results = self.retrieval_with_embedding(new_face_queries, new_non_face_queries, k, False)

      return results

    return results_with_face, D_with_face, results_without_face, D_without_face
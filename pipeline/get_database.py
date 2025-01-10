import os
import pickle


def get_database(feature_with_face, feature_without_face, general, remove_type, project_path, film):
  film_path = os.path.join(project_path, 'Group_Duy_Thang', 'index', film)
  database_with_face_path = os.path.join(film_path, 'with face', f'{feature_with_face}_{remove_type}.pkl')

  if general:
    database_without_face_path = os.path.join(film_path, 'without face', f'{feature_without_face}_s_{remove_type}.pkl' if feature_without_face == 'BEiT' else f'{feature_without_face}_rn_{remove_type}.pkl')
  else:
    database_without_face_path = os.path.join(film_path, 'without face', f'{feature_without_face}_{remove_type}.pkl')

  print(database_with_face_path)
  print(database_without_face_path)

  with open(database_with_face_path, 'rb') as f:
    database_with_face = pickle.load(f)
    index_with_face = database_with_face["index"]
    map_index_with_face = database_with_face["map_index"]

  map_index_with_face = [filename.replace('.npy', '') for filename in map_index_with_face]
  # map_index_with_face = {i: value for i, value in enumerate(map_index_with_face)}
  # map_index_with_face[-1] = 'No shot'

  with open(database_without_face_path, 'rb') as f:
    database_without_face = pickle.load(f)
    index_without_face = database_without_face["index"]
    map_index_without_face = database_without_face["map_index"]

  map_index_without_face = [filename.replace('.npy', '') for filename in map_index_without_face]
  # map_index_without_face = {i: value for i, value in enumerate(map_index_without_face)}
  # map_index_without_face[-1] = 'No shot'

  return index_with_face, map_index_with_face, index_without_face, map_index_without_face

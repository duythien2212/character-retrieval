from pipeline.get_input_features import get_input_features
from pipeline.get_database import get_database
from extract_feature.normalize_data import normalize_data
import numpy as np

class Retrieval:
    def __init__(self, list_query, feature_with_face="Facenet", feature_without_face="BEiT", remove_type="Compare", general=True, project_path="/content/drive/MyDrive/TRECVID", film="like_me"):
        self.input_features_with_face, self.input_features_without_face = get_input_features(list_query, feature_with_face, feature_without_face, general=general)
        self.index_with_face, self.map_index_with_face, self.index_without_face, self.map_index_without_face = get_database(feature_with_face, feature_without_face, general, remove_type, project_path, film)

    def search_with_face(self, k):
        if self.input_features_with_face.shape[0] > 0:
            print("Search with face")
            D_with_face, I_with_face = self.index_with_face.search(normalize_data(self.input_features_with_face), k)     # actual search
            print(I_with_face)                   # neighbors

            # Chuyển index thành tên shot
            results_with_face = []
            for query_results in I_with_face:
                # For each query, create a new list to store the mapped results
                mapped_results = []
                for result in query_results:
                    mapped_results.append(self.map_index_with_face[result] if result > 0 else 'No shot')
                results_with_face.append(mapped_results)
            results_with_face = np.array(results_with_face)
            print(results_with_face)
        else:
            D_with_face, I_with_face = None, None
            results_with_face = None
        
        return D_with_face, I_with_face, results_with_face

    def search_without_face(self, k):
        # Search without face
        if self.input_features_without_face.shape[0] > 0:
            print("Search without face")                      # we want to see 4 nearest neighbors
            D_without_face, I_without_face = self.index_without_face.search(normalize_data(self.input_features_without_face), k)     # actual search
            print(I_without_face)                   # neighbors

            # Chuyển index thành tên shot
            results_without_face = []
            for query_results in I_without_face:
                # For each query, create a new list to store the mapped results
                mapped_results = []
                for result in query_results:
                    mapped_results.append(self.map_index_without_face[result] if result > 0 else 'No shot')
                results_without_face.append(mapped_results)
            results_without_face = np.array(results_without_face)
            print(results_without_face)
        else:
            D_without_face, I_without_face = None, None
            results_without_face = None
        
        return D_without_face, I_without_face, results_without_face
from merge.merge_result import MergeResult


class MergeByCounter(MergeResult):
  def __init__(self, face_threshold=0.75, non_face_threshold=0.75):
    super().__init__(face_threshold, non_face_threshold)
    self.name = "counter"

  def cout_per_feature_type(self, result, D, threshold):
    relevant_items = dict()
    for query in range(len(result)):
      for i in range(len(result[query])):
        if D[query][i] > threshold:
          if result[query][i] not in relevant_items:
            relevant_items[result[query][i]] = 1
          else:
            relevant_items[result[query][i]] += 1
    return relevant_items

  def merge(self, results_with_face, results_without_face, D_with_face, D_without_face):
    cnt = dict()
    face_count = self.cout_per_feature_type(results_with_face, D_with_face, self.face_threshold)
    non_face_count = self.cout_per_feature_type(results_without_face, D_without_face, self.non_face_threshold)

    for key in set(face_count.keys()).union(set(non_face_count.keys())):
      cnt[key] = face_count.get(key, 0) + non_face_count.get(key, 0)

    sorted_cnt = dict(sorted(cnt.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_cnt.keys())
from merge.merge_result import MergeResult


class MergeByRank(MergeResult):
  def __init__(self, face_threshold=0.75, non_face_threshold=0.75):
    super().__init__(face_threshold, non_face_threshold)
    self.name = "rank"

  def merge(self, results_with_face, results_without_face, D_with_face, D_without_face):
    rm_dup_face_result = self.remove_duplicate(zip(results_with_face, D_with_face))
    rm_dup_non_face_result = self.remove_duplicate(zip(results_without_face, D_without_face))

    sorted_with_face = self.sort_result(rm_dup_face_result)
    sorted_without_face = self.sort_result(rm_dup_non_face_result)

    merged_results = dict()
    for i in range(len(sorted_with_face)):
      merged_results[sorted_with_face[i][0]] = 1/(i+1)

    for i in range(len(sorted_without_face)):
      if sorted_without_face[i][0] not in merged_results:
        merged_results[sorted_without_face[i][0]] = 1/(i+1)
      else:
        merged_results[sorted_without_face[i][0]] += 1/(i+1)

    sorted_merged_results = dict(sorted(merged_results.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_merged_results.keys())
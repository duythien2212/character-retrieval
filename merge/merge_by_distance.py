from merge.merge_result import MergeResult


class MergeByDistance(MergeResult):
  def __init__(self, face_threshold=0.75, non_face_threshold=0.75):
    super().__init__(face_threshold, non_face_threshold)
    self.name = "distance"

  def merge(self, results_with_face, results_without_face, D_with_face, D_without_face):
    rm_dup_face_result = self.remove_duplicate(zip(results_with_face, D_with_face))
    rm_dup_non_face_result = self.remove_duplicate(zip(results_without_face, D_without_face))

    sorted_result = self.sort_result(rm_dup_face_result + rm_dup_non_face_result)

    sorted_result_name = [item[0] for item in sorted_result]
    sorted_result_distance = [item[1] for item in sorted_result]

    sorted_result = self.remove_duplicate(zip([sorted_result_name], [sorted_result_distance]))

    return [item[0] for item in sorted_result]
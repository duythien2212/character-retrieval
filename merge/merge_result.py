from abc import abstractmethod


class MergeResult():
  def __init__(self, face_threshold = 0.75, non_face_threshold = 0.75):
    self.face_threshold = face_threshold
    self.non_face_threshold = non_face_threshold

  @abstractmethod
  def merge(self, results_with_face, results_without_face, D_with_face, D_without_face):
    pass

  def remove_duplicate(self, result):
    rm_dup_result = dict()
    result = list(result)
    for query in range(len(result)):
      for i in range(len(result[query][0])):
        item = (result[query][0][i], result[query][1][i])
        if (item[0] not in rm_dup_result.keys()):
          rm_dup_result[item[0]] = item[1]
        else:
          rm_dup_result[item[0]] = max(rm_dup_result[item[0]], item[1])

    return list(rm_dup_result.items())

  def sort_result(self, result):
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result
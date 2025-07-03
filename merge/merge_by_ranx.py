from ranx import Run, fuse

from merge.merge_result import MergeResult


class MergeByRanx(MergeResult):
  def __init__(self, face_threshold=0.75, non_face_threshold=0.75):
    super().__init__(face_threshold, non_face_threshold)
    self.name = "ranx"

  def merge(self, results_with_face, results_without_face, D_with_face, D_without_face):
    runs = []

    result = results_with_face + results_without_face
    distance = []
    for i in range(len(results_with_face)):
      distance.append(D_with_face[i])
    for i in range(len(results_without_face)):
      distance.append(D_without_face[i])

    for i, (res, sims) in enumerate(zip(result, distance)):
      query_id = f"query_0"
      run_data = {str(doc_id): score for doc_id, score in zip(res, sims)}
      run = Run({query_id: run_data})
      runs.append(run)
      # Hợp nhất kết quả
    fused_run = fuse(runs, method='rrf')  # Có thể điều chỉnh weights nếu cần
    fused_ranking_results = {}

    for query_id, docs in fused_run.to_dict().items():
      fused_ranking_results[query_id] = list(docs.keys())  # Lấy danh sách document IDs (không cần điểm số)
    return fused_ranking_results['query_0']

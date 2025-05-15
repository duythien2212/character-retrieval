import numpy as np
from ranx import Run, fuse

def merge_sort_by_ranx(results, D, method='rrf'):
  # Chuyển đổi dữ liệu sang dạng dictionary
  runs = []
  for i, (res, sims) in enumerate(zip(results, D)):
      query_id = f"query_0"
      run_data = {str(doc_id): score for doc_id, score in zip(res, sims)}
      run = Run({query_id: run_data})
      runs.append(run)
  # Hợp nhất kết quả
  fused_run = fuse(runs, method=method)  # Có thể điều chỉnh weights nếu cần

  fused_ranking_results = {}

  for query_id, docs in fused_run.to_dict().items():
      fused_ranking_results[query_id] = list(docs.keys())  # Lấy danh sách document IDs (không cần điểm số)

  return fused_ranking_results['query_0']

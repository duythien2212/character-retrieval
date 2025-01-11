import numpy as np

def merge_sort_by_distance(results, D):
  combined = []
  for query_distances, query_indices in zip(D, results):
      combined.extend(zip(query_distances, query_indices))

    # Sắp xếp dựa trên giá trị khoảng cách (D) theo thứ tự giảm dần
  combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)

  # Lọc để giữ lại giá trị có D cao nhất cho mỗi chỉ số trong I
  filtered_indices = {}
  for distance, index in combined_sorted:
      if index not in filtered_indices:  # Chỉ thêm nếu chưa tồn tại
          filtered_indices[index] = distance

  # Kết quả cuối cùng: chỉ số đã lọc
  result = list(filtered_indices.keys())

  return result
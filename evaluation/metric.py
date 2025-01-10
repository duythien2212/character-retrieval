import numpy as np

def precision(true_positives, retrieved):
  return true_positives / retrieved if retrieved != 0 else 0

def recall(true_positives, relevant):
  return true_positives / relevant if relevant != 0 else 0

def f1_score(precision, recall):
  return 2 * precision * recall / (precision + recall)

def average_precision(relevant_items, retrieved_items):
  '''
  relevant_items: list các ID mà thực sự là đúng
  retrieved_items: list các ID được truy vấn trả về
  '''
  relevant_items = set(relevant_items)
  retrieved = 0
  true_positives = 0
  ap = 0

  for i, item in enumerate(retrieved_items):
    retrieved += 1
    if item in relevant_items:
      true_positives += 1
      ap += precision(true_positives, retrieved)

  return ap / len(relevant_items) if relevant_items else 0

def mean_average_precision(queries):
  aps = [average_precision(q[0], q[1]) for q in queries]
  return np.mean(aps) if aps else 0

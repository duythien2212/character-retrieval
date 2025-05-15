import numpy as np
from collections import Counter

def merge_sort_by_counter(results):
  merged_results = results.flatten()
  counts = Counter(merged_results)
  merged_results = sorted(counts, key=counts.get, reverse=True)
  return merged_results

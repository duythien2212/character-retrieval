import numpy as np
from merge.merge_by_distance import merge_sort_by_distance
from merge.merge_by_counter import merge_sort_by_counter
from merge.merge_by_ranx import merge_sort_by_ranx

def merge_result(results_with_face, results_without_face, D_with_face, D_without_face, type='no'):
  if type == 'no':
    no_merge_result_1 = list(set(results_with_face.flatten())) if results_with_face is not None else []
    no_merge_result_2 = list(set(results_without_face.flatten())) if results_without_face is not None else []
    no_merge_result = no_merge_result_1 + no_merge_result_2
    no_merge_result = list(set(no_merge_result))
    no_merge_result = [x for x in no_merge_result if x != 'No shot']
    return no_merge_result

  elif type == 'Distance':
    if results_with_face is None and results_without_face is None:
      return []
    if results_with_face is None:
      results = merge_sort_by_distance(results_without_face, D_without_face)
    elif results_without_face is None:
      results = merge_sort_by_distance(results_with_face, D_with_face)
    else:
      results = merge_sort_by_distance(np.concatenate((results_with_face, results_without_face)), np.concatenate((D_with_face, D_without_face)))
    result = [x for x in results if x != 'No shot']
    return result

  elif type == 'Counter':
    if results_with_face is None and results_without_face is None:
      return []
    if results_with_face is None:
      results = merge_sort_by_counter(results_without_face)
    elif results_without_face is None:
      results = merge_sort_by_counter(results_with_face)
    else:
      results = merge_sort_by_counter(np.concatenate((results_with_face, results_without_face)))
    result = [x for x in results if x != 'No shot']
    return result

  elif type == 'Ranx':
    if results_with_face is None and results_without_face is None:
      return []
    method = 'rrf'
    if results_with_face is None:
      results = merge_sort_by_ranx(results_without_face, D_without_face, method)
    elif results_without_face is None:
      results = merge_sort_by_ranx(results_with_face, D_with_face, method)
    else:
      results = merge_sort_by_ranx(np.concatenate((results_with_face, results_without_face)), np.concatenate((D_with_face, D_without_face)), method)
    result = [x for x in results if x != 'No shot']
    return result

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

class Evaluation():
  def __init__(self, retrieval_results, ground_truth):
    self.retrieval_results = retrieval_results
    self.ground_truth = ground_truth
    tmp = dict()

    for query in range(len(self.retrieval_results)):
      for name in self.retrieval_results[query]:
        if name not in tmp.keys():
          tmp[name] = 1
        else:
          raise ValueError(f"Duplicate shot: {name}!")

  def calc_true_positives(self, query):
    true_positives = [1 if item in self.ground_truth else 0 for item in self.retrieval_results[query]]
    return true_positives

  def precision(self, query):
    true_positives = self.calc_true_positives(query)
    precision = np.sum(true_positives) / len(self.retrieval_results) if len(self.retrieval_results) > 0 else 0
    return precision

  def recall(self, query):
    true_positives = self.calc_true_positives(query)
    recall = np.sum(true_positives) / len(self.ground_truth) if len(self.ground_truth) > 0 else 0
    return recall

  def average_precision(self, query):
    ap = 0
    count_tf = 0
    true_positives = self.calc_true_positives(query)
    for i, item in enumerate(true_positives):
      if item == 1:
        count_tf += 1
        ap += count_tf / (i + 1)
    return ap / len(self.ground_truth) if len(self.ground_truth) > 0 else 0

  def mean_average_precision(self):
    aps = [self.average_precision(query) for query in range(len(self.retrieval_results))]
    return np.mean(aps) if aps else 0

  def plot_precision_recall_curve(self):
    plt.figure(figsize=(10, 6))
    for query in range(len(self.retrieval_results)):
      true_positives = self.calc_true_positives(query)
      precision_vals = []
      recall_vals = []
      count_tf = 0
      for i, item in enumerate(true_positives):
        if item == 1:
          count_tf += 1
        p = count_tf / (i + 1)
        r = count_tf / len(self.ground_truth)
        precision_vals.append(p)
        recall_vals.append(r)

      auc_score = auc(recall_vals, precision_vals)
      plt.plot(recall_vals, precision_vals, label=f'AUC = {auc_score:.2f}')

    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.show()

  def plot_ap_with_map(self):
    aps = [self.average_precision(query) for query in range(len(self.retrieval_results))]
    map_score = self.mean_average_precision()

    plt.figure(figsize=(10, 6))

    x = np.arange(len(aps))

    plt.bar(x, aps, color='blue', alpha=0.7, label='Average Precision (AP)')
    plt.axhline(y=map_score, color='red', linestyle='--', label=f'mAP = {map_score:.2f}')

    plt.title('Average Precision (AP) and Mean Average Precision (mAP)')
    plt.xlabel('Queries')
    plt.ylabel('Average Precision')
    plt.xticks(x, [f'Query {i+1}' for i in range(len(aps))])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

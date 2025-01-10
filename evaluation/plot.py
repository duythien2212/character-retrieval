import matplotlib.pyplot as plt

# Hàm vẽ Precision-Recall Curve
def plot_precision_recall_curve(queries, tittle='Precision-Recall Curve'):
    plt.figure(figsize=(10, 6))

    for idx, (relevant_items, retrieved_items) in enumerate(queries):
        relevant_items = set(relevant_items)
        precision_vals = []
        recall_vals = []
        true_positives = 0

        # Tính Precision và Recall cho từng ngưỡng
        for i, item in enumerate(retrieved_items):
            if item in relevant_items:
                true_positives += 1
            p = precision(true_positives, i + 1)
            r = recall(true_positives, len(relevant_items))
            precision_vals.append(p)
            recall_vals.append(r)

        auc_score = auc(recall_vals, precision_vals)

        # Vẽ đường Precision-Recall cho từng truy vấn
        plt.plot(recall_vals, precision_vals, label=f'AUC = {auc_score:.2f}')

    # Cấu hình biểu đồ
    plt.title(tittle)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.show()

# Hàm vẽ biểu đồ AP và mAP
def plot_ap_with_map(queries):
    # Tính AP cho từng truy vấn
    aps = [average_precision(q[0], q[1]) for q in queries]
    map_score = mean_average_precision(queries)

    plt.figure(figsize=(10, 6))

    # Vẽ cột AP
    x = np.arange(len(aps))  # Vị trí các cột
    plt.bar(x, aps, color='blue', alpha=0.7, label='Average Precision (AP)')

    # Vẽ đường mAP
    plt.axhline(y=map_score, color='red', linestyle='--', label=f'mAP = {map_score:.2f}')

    # Cấu hình biểu đồ
    plt.title('Average Precision (AP) and Mean Average Precision (mAP)')
    plt.xlabel('Queries')
    plt.ylabel('Average Precision')
    plt.xticks(x, [f'Query {i+1}' for i in range(len(aps))])  # Nhãn các query
    plt.ylim(0, 1)  # AP nằm trong khoảng [0, 1]
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def merge_sort(results):
  merged_results = results.flatten()
  counts = Counter(merged_results)
  merged_results = sorted(counts, key=counts.get, reverse=True)
  return merged_results
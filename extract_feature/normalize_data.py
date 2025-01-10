import numpy as np

def normalize_data(data):
    # Chuẩn hóa dữ liệu
    normalized_data = data / np.linalg.norm(data, axis=1, keepdims=True)
    return normalized_data

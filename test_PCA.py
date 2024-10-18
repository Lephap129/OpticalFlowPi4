import numpy as np

# Tạo dữ liệu cho chuyển động tịnh tiến
translation_vector = np.array([2, 1])

# Tạo dữ liệu cho chuyển động quay
angle = np.linspace(0, 2*np.pi, 10)
radius = 5
vectors_rotation = np.array([
    [radius * np.cos(a), radius * np.sin(a)] for a in angle
])

# Kết hợp chuyển động tịnh tiến và chuyển động quay
vectors_combined = vectors_rotation + translation_vector

def pca_manual(vectors):
    # Chuẩn hóa dữ liệu
    mean_vector = np.mean(vectors, axis=0)
    V_norm = vectors - mean_vector

    # Tính toán ma trận hiệp phương sai
    covariance_matrix = np.cov(V_norm, rowvar=False)

    # Tính toán các giá trị riêng (eigenvalues) và vector riêng (eigenvectors)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sắp xếp các giá trị riêng và vector riêng theo thứ tự giảm dần của các giá trị riêng
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvectors, eigenvalues, mean_vector

def analyze_vectors(vectors, title):
    # Tính độ dịch chuyển tổng thể
    translation_vector = np.mean(vectors, axis=0)

    # Thực hiện PCA thủ công
    components, explained_variance, mean_vector = pca_manual(vectors)

    # Tính góc quay từ các thành phần chính
    angle_of_rotation = np.arctan2(components[1, 1], components[1, 0])

    print(f"Kết quả phân tích cho {title}:")
    print(f"Độ dịch chuyển tổng thể: {translation_vector}")
    print(f"Góc quay (rad): {angle_of_rotation}")
    print(f"Góc quay (deg): {np.degrees(angle_of_rotation)}")
    print(f"Các thành phần chính:\n", components)
    print(f"Giá trị riêng (explained variance):\n", explained_variance)

# Phân tích chuyển động kết hợp
analyze_vectors(vectors_combined, "Chuyển động kết hợp (xoay và tịnh tiến)")
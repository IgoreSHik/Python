import numpy as np

mean_vector = np.array([1, 2, 3])
covariance_matrix = np.array([[1, 0.5, 0.2],
                              [0.5, 2, 0.8],
                              [0.2, 0.8, 1]])

num_samples = 1000

generated_data = np.random.multivariate_normal(mean_vector, covariance_matrix, num_samples)

print("Wygenerowane dane:")
print(generated_data)
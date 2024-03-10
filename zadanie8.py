import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

mean_class1 = np.array([1, 2])
covariance_matrix_class1 = np.array([[1, 0.5],
                                     [0.5, 2]])

mean_class2 = np.array([4, 5])
covariance_matrix_class2 = np.array([[1, -0.5],
                                     [-0.5, 3]])

num_samples_per_class = 100
data_class1 = np.random.multivariate_normal(mean_class1, covariance_matrix_class1, num_samples_per_class)
data_class2 = np.random.multivariate_normal(mean_class2, covariance_matrix_class2, num_samples_per_class)

X = np.vstack((data_class1, data_class2))
y = np.concatenate((np.zeros(num_samples_per_class), np.ones(num_samples_per_class)))

# Dwuklasowy klasyfikator liniowy
lr_classifier = LogisticRegression()
lr_probs = cross_val_predict(lr_classifier, X, y, cv=StratifiedKFold(n_splits=5))

# Klasyfikator minimalnej odległości k-NN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_probs = cross_val_predict(knn_classifier, X, y, cv=StratifiedKFold(n_splits=5))

fpr_lr, tpr_lr, _ = roc_curve(y, lr_probs)
fpr_knn, tpr_knn, _ = roc_curve(y, knn_probs)

roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Krzywa ROC dla klasyfikatora liniowego (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_knn, tpr_knn, color='red', lw=2, label=f'Krzywa ROC dla klasyfikatora k-NN (AUC = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Losowe zgadywanie')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywe ROC dla klasyfikatora liniowego i k-NN')
plt.legend(loc='lower right')
plt.show()
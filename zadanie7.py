import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats import multivariate_normal

# dla dwóch klas
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bayesian_clf = GaussianNB()
bayesian_clf.fit(X_train, y_train)

qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)

bayesian_preds = bayesian_clf.predict(X_test)
qda_preds = qda_clf.predict(X_test)

bayesian_accuracy = accuracy_score(y_test, bayesian_preds)
qda_accuracy = accuracy_score(y_test, qda_preds)

print(f"Dokładność klasyfikatora bayesowskiego: {bayesian_accuracy:.2f}")
print(f"Dokładność kwadratowego analizatora dyskryminacyjnego: {qda_accuracy:.2f}")

print("\nRaport klasyfikacji klasyfikatora bayesowskiego:")
print(classification_report(y_test, bayesian_preds))

print("\nRaport kwadratowego analizatora dyskryminacyjnego:")
print(classification_report(y_test, qda_preds))

bayesian_probs = bayesian_clf.predict_proba(X_test)[:, 1]
qda_probs = qda_clf.predict_proba(X_test)[:, 1]

fpr_bayesian, tpr_bayesian, _ = roc_curve(y_test, bayesian_probs)
fpr_qda, tpr_qda, _ = roc_curve(y_test, qda_probs)

roc_auc_bayesian = auc(fpr_bayesian, tpr_bayesian)
roc_auc_qda = auc(fpr_qda, tpr_qda)

plt.figure(figsize=(8, 6))
plt.plot(fpr_bayesian, tpr_bayesian, color='blue', lw=2, label=f'Krzywa ROC klasyfikatora bayesowskiego (AUC = {roc_auc_bayesian:.2f})')
plt.plot(fpr_qda, tpr_qda, color='red', lw=2, label=f'Krzywa ROC kwadratowego analizatora dyskryminacyjnego (AUC = {roc_auc_qda:.2f})')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Losowe zgadywanie')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Dwie klassy')
plt.legend(loc='lower right')
plt.show()



# dla trzech klas
mean_class3 = np.array([7, 8])
covariance_matrix_class3 = np.array([[2, 0],
                                     [0, 1]])

num_samples_per_class = 100
data_class1 = np.random.multivariate_normal(mean_class1, covariance_matrix_class1, num_samples_per_class)
data_class2 = np.random.multivariate_normal(mean_class2, covariance_matrix_class2, num_samples_per_class)
data_class3 = np.random.multivariate_normal(mean_class3, covariance_matrix_class3, num_samples_per_class)

X = np.vstack((data_class1, data_class2, data_class3))
y = np.concatenate((np.zeros(num_samples_per_class), np.ones(num_samples_per_class), 2 * np.ones(num_samples_per_class)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bayesian_clf = GaussianNB()
bayesian_clf.fit(X_train, y_train)

qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)

bayesian_preds = bayesian_clf.predict(X_test)
qda_preds = qda_clf.predict(X_test)

bayesian_accuracy = accuracy_score(y_test, bayesian_preds)
qda_accuracy = accuracy_score(y_test, qda_preds)

print(f"Dokładność klasyfikatora bayesowskiego: {bayesian_accuracy:.2f}")
print(f"Dokładność kwadratowego analizatora dyskryminacyjnego: {qda_accuracy:.2f}")

print("\nRaport klasyfikacji klasyfikatora bayesowskiego:")
print(classification_report(y_test, bayesian_preds))

print("\nRaport kwadratowego analizatora dyskryminacyjnego:")
print(classification_report(y_test, qda_preds))

bayesian_probs = bayesian_clf.predict_proba(X_test)
qda_probs = qda_clf.predict_proba(X_test)

y_one_hot = np.eye(3)[y_test.astype(int)]

fpr_bayesian = dict()
tpr_bayesian = dict()
fpr_qda = dict()
tpr_qda = dict()
roc_auc_bayesian = dict()
roc_auc_qda = dict()

for i in range(3):
    fpr_bayesian[i], tpr_bayesian[i], _ = roc_curve(y_one_hot[:, i], bayesian_probs[:, i])
    fpr_qda[i], tpr_qda[i], _ = roc_curve(y_one_hot[:, i], qda_probs[:, i])

    roc_auc_bayesian[i] = auc(fpr_bayesian[i], tpr_bayesian[i])
    roc_auc_qda[i] = auc(fpr_qda[i], tpr_qda[i])

plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']

for i in range(3):
    plt.plot(fpr_bayesian[i], tpr_bayesian[i], color=colors[i], lw=2,
             label=f'Krzywa ROC klasyfikatora bayesowskiego (AUC klasa {i} = {roc_auc_bayesian[i]:.2f})')

    plt.plot(fpr_qda[i], tpr_qda[i], color=colors[i], linestyle='--', lw=2,
             label=f'Krzywa ROC kwadratowego analizatora dyskryminacyjnego (AUC klasa {i} = {roc_auc_qda[i]:.2f})')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Losowe zgadywanie')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Trzy klassy')
plt.legend(loc='lower right')
plt.show()
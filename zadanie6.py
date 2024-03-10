import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

#Klasyfikator dla dwoch klass
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

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Dwie klassy accuracy: {accuracy:.2f}")

print("\nDwie klassy report:")
print(classification_report(y_test, y_pred))

#Klasyfikator dla trzech klass
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

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Trzy klassy accuracy: {accuracy:.2f}")

print("\nTrzy klassy report:")
print(classification_report(y_test, y_pred))
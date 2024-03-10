import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

num_iterations = 100
num_samples = 10000
num_features = 50

id3_times, c45_times, cart_times = [], [], []
id3_sizes, c45_sizes, cart_sizes = [], [], []
id3_accuracies, c45_accuracies, cart_accuracies = [], [], []

for _ in range(num_iterations):
    X, y = make_classification(n_samples=num_samples, n_features=num_features, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ID3
    start_time_id3 = time.time()
    id3_model = DecisionTreeClassifier(criterion='entropy')
    id3_model.fit(X_train, y_train)
    id3_predictions = id3_model.predict(X_test)
    id3_accuracy = accuracy_score(y_test, id3_predictions)
    id3_tree_generation_time = time.time() - start_time_id3

    id3_times.append(id3_tree_generation_time)
    id3_sizes.append(id3_model.tree_.node_count)
    id3_accuracies.append(id3_accuracy)

    # C4.5
    start_time_c45 = time.time()
    c45_model = DecisionTreeClassifier(criterion='entropy', splitter='best')
    c45_model.fit(X_train, y_train)
    c45_predictions = c45_model.predict(X_test)
    c45_accuracy = accuracy_score(y_test, c45_predictions)
    c45_tree_generation_time = time.time() - start_time_c45

    c45_times.append(c45_tree_generation_time)
    c45_sizes.append(c45_model.tree_.node_count)
    c45_accuracies.append(c45_accuracy)

    # CART
    start_time_cart = time.time()
    cart_model = DecisionTreeClassifier(criterion='gini')
    cart_model.fit(X_train, y_train)
    cart_predictions = cart_model.predict(X_test)
    cart_accuracy = accuracy_score(y_test, cart_predictions)
    cart_tree_generation_time = time.time() - start_time_cart

    cart_times.append(cart_tree_generation_time)
    cart_sizes.append(cart_model.tree_.node_count)
    cart_accuracies.append(cart_accuracy)

print(f'ID3: Average Time: {np.mean(id3_times):.4f} seconds, Average Size: {np.mean(id3_sizes)} nodes, Average Accuracy: {np.mean(id3_accuracies):.4f}')
print(f'C4.5: Average Time: {np.mean(c45_times):.4f} seconds, Average Size: {np.mean(c45_sizes)} nodes, Average Accuracy: {np.mean(c45_accuracies):.4f}')
print(f'CART: Average Time: {np.mean(cart_times):.4f} seconds, Average Size: {np.mean(cart_sizes)} nodes, Average Accuracy: {np.mean(cart_accuracies):.4f}')
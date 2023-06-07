import numpy as np
from collections import Counter

# Distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    # The number of neighbours to be considered
    def __init__(self, k=3):
        self.k = k
    
    # Training sample and label
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # This can have multiple samples
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        # Compute distances (nearest neighbour). 
        # Find the distance between new sample and train sample
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get k nearest samples, labels
        k_indicies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indicies]
        # Majority vote, most common class label. 
        # Counter finds the most common value and the number of times it occurs.
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


    
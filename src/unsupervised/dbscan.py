import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN as SklearnDBSCAN

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', p=3):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.p = p
        self.labels = None

    def _calculate_distance(self, p1, p2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((p1 - p2)**2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(p1 - p2))
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.power(np.abs(p1 - p2), self.p)), 1/self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _get_neighbors(self, X, point_index):
        neighbors = []
        for i in range(len(X)):
            if self._calculate_distance(X[point_index], X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, labels, point_index, neighbors, cluster_id):
        labels[point_index] = cluster_id

        i = 0
        while i < len(neighbors):
            current_point_idx = neighbors[i]

            if labels[current_point_idx] == -1: # noise
                labels[current_point_idx] = cluster_id
            
            elif labels[current_point_idx] == -2: # unvisited
                labels[current_point_idx] = cluster_id
                
                new_neighbors = self._get_neighbors(X, current_point_idx)
                
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            
            i += 1

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        labels = np.full(n_samples, -2) 
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != -2:
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1
        
        self.labels = labels
        return self

def main():
    print("DBSCAN Clustering")
    csv_path = input("Enter path to your CSV data file: ")
    df = pd.read_csv(csv_path)
        
    print("Columns in your data:", list(df.columns))

    feature_cols_str = input("Enter the feature columns (separated by comma): ")
    feature_cols = [col.strip() for col in feature_cols_str.split(',')]
    
    X_df = df[feature_cols]
    X_numeric = X_df.select_dtypes(include=np.number)
    X = X_numeric.values

    eps = float(input("Enter the epsilon value (ex 0.5): ") or 0.5)
    min_samples = int(input("Enter the minimum number of samples (ex 5): ") or 5)
    metric = input("Enter the distance metric (euclidean/manhattan/minkowski): ").lower() or 'euclidean'
    
    p_minkowski = 3
    if metric == 'minkowski':
        p_minkowski = int(input("Enter the 'p' for Minkowski distance (ex 3): ") or 3)
    elif metric not in ['euclidean', 'manhattan']:
        print("Metric not recognized. Defaulting to 'euclidean'.")
        metric = 'euclidean'

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("="*40)
    # Non-sklearn DBSCAN Implementation
    print("Using non-sklearn DBSCAN implementation...")
    
    custom_model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, p=p_minkowski)
    custom_model.fit(X_scaled)
    
    custom_labels = custom_model.labels
    n_clusters_custom = len(set(custom_labels)) - (1 if -1 in custom_labels else 0)
    n_noise_custom = list(custom_labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters_custom}")
    print(f"Estimated number of noise points: {n_noise_custom}")

    df['custom_cluster'] = custom_labels
    print("\nData with Cluster Labels (Custom):")
    print(df.head())

    print("="*40)
    # Sklearn DBSCAN Implementation
    print("Using sklearn DBSCAN implementation...")
    
    sklearn_model = SklearnDBSCAN(eps=eps, min_samples=min_samples, metric=metric, p=p_minkowski)
    sklearn_model.fit(X_scaled)

    sklearn_labels = sklearn_model.labels_
    n_clusters_sklearn = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
    n_noise_sklearn = list(sklearn_labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters_sklearn}")
    print(f"Estimated number of noise points: {n_noise_sklearn}")

    df['sklearn_cluster'] = sklearn_labels
    print("\nData with Cluster Labels (Sklearn):")
    print(df.head())

if __name__ == "__main__":
    main()


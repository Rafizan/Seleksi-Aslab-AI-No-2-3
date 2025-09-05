import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SklearnKMeans

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, init_method='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init_method = init_method
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))
    
    def _calculate_inertia(self, X):
        """Calculates the within-cluster sum of squares (inertia)."""
        inertia = 0
        for i in range(self.n_clusters):
            points_in_cluster = X[self.labels == i]
            if len(points_in_cluster) > 0:
                inertia += np.sum((points_in_cluster - self.centroids[i])**2)
        return inertia

    def _initialize_centroids(self, X):
        if self.init_method == 'random':
            random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[random_indices]
        elif self.init_method == 'kmeans++':
            centroids = [X[np.random.randint(X.shape[0])]]
            for _ in range(1, self.n_clusters):
                distances = []
                for data_point in X:
                    min_dist_sq = min([self._euclidean_distance(data_point, c)**2 for c in centroids])
                    distances.append(min_dist_sq)
                probabilities = np.array(distances) / np.sum(distances)
                cumulative_probabilities = np.cumsum(probabilities)
                r = np.random.rand()

                new_centroid_idx = np.searchsorted(cumulative_probabilities, r)
                centroids.append(X[new_centroid_idx])
            self.centroids = np.array(centroids)

    def _assign_clusters(self, X):
        labels = []
        for point in X:
            distances_to_centroids = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
            cluster_label = np.argmin(distances_to_centroids)
            labels.append(cluster_label)
        return np.array(labels)

    def _update_centroids(self, X, labels):
        new_centroids = []
        for i in range(self.n_clusters):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                new_centroid = np.mean(points_in_cluster, axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(self.centroids[i])
        return np.array(new_centroids)

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self._initialize_centroids(X)
        
        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()
            labels = self._assign_clusters(X)
            self.centroids = self._update_centroids(X, labels)
            if np.allclose(old_centroids, self.centroids):
                break
        
        self.labels = self._assign_clusters(X)
        self.inertia_ = self._calculate_inertia(X)


    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._assign_clusters(X)

def main():
    print("K-Means Clustering")
    csv_path = input("Enter path to your CSV data file: ")
    df = pd.read_csv(csv_path)
    print("Columns in your data:", list(df.columns))

    feature_cols_str = input("Enter the feature columns (separated by comma): ")
    feature_cols = [col.strip() for col in feature_cols_str.split(',')]
    
    X_df = df[feature_cols]
    X_numeric = X_df.select_dtypes(include=np.number)
    X = X_numeric.values

    if X.shape[1] == 0:
        print("Error: No numeric feature columns selected. Please select columns with numerical data.")
        return

    n_clusters = int(input("Enter the number of clusters (ex 3): ") or 3)
    max_iter = int(input("Enter the maximum number of iterations (ex 100): ") or 100)
    init_method = input("Enter the initialization method (random/kmeans++): ") or 'random'
    if init_method not in ['random', 'kmeans++']:
        print("Method not recognized. Using 'random'.")
        init_method = 'random'

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("="*40)
    # Non-sklearn K-Means
    print("Using custom K-Means implementation...")
    
    custom_model = KMeans(n_clusters=n_clusters, max_iter=max_iter, init_method=init_method)
    custom_model.fit(X_scaled)
    
    print("Centroid Final (Custom):")
    print(scaler.inverse_transform(custom_model.centroids))
    print(f"Inertia (Sum of Squared Errors): {custom_model.inertia_:.4f}")

    df['custom_cluster'] = custom_model.labels
    print("\nData with Cluster Labels (Custom):")
    print(df.head())

    print("="*40)
    # Sklearn K-Means
    print("Using sklearn K-Means implementation...")
    
    sklearn_init = 'random' if init_method == 'random' else 'k-means++'
    sklearn_model = SklearnKMeans(n_clusters=n_clusters, max_iter=max_iter, init=sklearn_init, n_init=1, random_state=42)
    sklearn_model.fit(X_scaled)

    print("Centroid Final (Sklearn):")
    print(scaler.inverse_transform(sklearn_model.cluster_centers_))
    print(f"Inertia (Sum of Squared Errors): {sklearn_model.inertia_:.4f}")

    df['sklearn_cluster'] = sklearn_model.labels_
    print("\nData with Cluster Labels (Sklearn):")
    print(df.head())

if __name__ == "__main__":
    main()
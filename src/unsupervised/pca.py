import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components_ = eigenvectors[0:self.n_components]
        
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[0:self.n_components] / total_variance

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def main():
    print("Principal Component Analysis (PCA)")
    csv_path = input("Enter path to your CSV data file: ")
    df = pd.read_csv(csv_path)
    print("Columns in your data:", list(df.columns))

    feature_cols_str = input("Enter the feature columns (separated by comma): ")
    feature_cols = [col.strip() for col in feature_cols_str.split(',')]
    
    X_df = df[feature_cols]
    X_numeric = X_df.select_dtypes(include=np.number)
    X = X_numeric.values

    n_components = int(input("Enter the number of principal components to keep (ex 2): ") or 2)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("="*40)
    # Non-sklearn PCA Implementation
    print("Using non-sklearn PCA implementation...")
    
    custom_pca = PCA(n_components=n_components)
    X_custom_transformed = custom_pca.fit_transform(X_scaled)
    
    print(f"Explained Variance Ratio (Custom): {custom_pca.explained_variance_ratio_}")
    print(f"Sum of Explained Variance (Custom): {np.sum(custom_pca.explained_variance_ratio_):.4f}")

    # Add transformed components to the DataFrame
    for i in range(n_components):
        df[f'custom_PC{i+1}'] = X_custom_transformed[:, i]

    print("\nData with Custom PCA Components:")
    print(df.head())


    print("="*40)
    # Sklearn PCA Implementation
    print("Using sklearn PCA implementation...")
    
    sklearn_pca = SklearnPCA(n_components=n_components)
    X_sklearn_transformed = sklearn_pca.fit_transform(X_scaled)
    
    print(f"Explained Variance Ratio (Sklearn): {sklearn_pca.explained_variance_ratio_}")
    print(f"Sum of Explained Variance (Sklearn): {np.sum(sklearn_pca.explained_variance_ratio_):.4f}")

    for i in range(n_components):
        df[f'sklearn_PC{i+1}'] = X_sklearn_transformed[:, i]

    print("\nData with Sklearn PCA Components:")
    print(df.head())


if __name__ == "__main__":
    main()

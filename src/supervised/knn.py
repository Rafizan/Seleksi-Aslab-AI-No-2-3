import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from scaler import Scaler, precision_score

def minkowski_distance(a, b, p):
    # Root is not used for efficiency
    return np.sum(np.abs(a - b) ** p)

def knn_predict(X_train, y_train, X_test, k=3, p=2):
    predictions = []
    for i in range(len(X_test)):
        distances = X_train.apply(lambda x: minkowski_distance(x.values, X_test.iloc[i].values, p), axis=1)
        k_indices = distances.argsort()[:k]
        k_nearest_labels = y_train.iloc[k_indices]
        pred = k_nearest_labels.mode()[0]
        predictions.append(pred)
    return np.array(predictions)

def main():
    print("KNN Prediction")

    csv_path = input("Enter path to your CSV data file: ")
    df = pd.read_csv(csv_path)
    print("Columns in your data:", list(df.columns))
    y_col = input("Enter the name of the target column: ")
    X = df.drop(columns=[y_col])
    y = df[y_col]

    print("Choose distance metric:")
    print("1. Euclidean (p=1)")
    print("2. Manhattan (p=2)")
    print("3. Minkowski (custom p)")
    choice = input("Enter choice (1/2/3): ")
    if choice == "1":
        p = 1
    elif choice == "2":
        p = 2
    else:
        p = float(input("Enter p value: "))

    k = int(input("Enter number of neighbors (k): "))

    print("="*40)
    # Non-sklearn KNN
    print("Using non-sklearn KNN implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    preds = knn_predict(X_train, y_train, X_test, k=k, p=p)
    precision = precision_score(y_test.values, preds)
    print(f"Precision using hold-out validation: {precision * 100:.2f}%")

    NSPLIT = 5
    precisions = np.zeros(NSPLIT)
    skf = StratifiedKFold(n_splits=NSPLIT, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = Scaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        preds = knn_predict(X_train, y_train, X_test, k=k, p=p)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

    print("="*40)
    print("Using sklearn KNN implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=k, p=p)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    precision = precision_score(y_test.values, preds)
    print(f"Precision using hold-out validation: {precision * 100:.2f}%")

    precisions = np.zeros(NSPLIT)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = Scaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = KNeighborsClassifier(n_neighbors=k, p=p)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

if __name__ == "__main__":
    main()
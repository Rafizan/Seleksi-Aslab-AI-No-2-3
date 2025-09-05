import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from scaler import Scaler, precision_score

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini(y):
    classes = np.unique(y)
    impurity = 1.0
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity -= p ** 2
    return impurity

def best_split(X, y):
    best_gini = 1.0
    best_feature = None
    best_threshold = None
    for feature in X.columns:
        thresholds = np.unique(X[feature])
        for t in thresholds:
            left_mask = X[feature] <= t
            right_mask = X[feature] > t
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            gini_left = gini(y[left_mask])
            gini_right = gini(y[right_mask])
            weighted_gini = (np.sum(left_mask) * gini_left + np.sum(right_mask) * gini_right) / len(y)
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = t
    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=5, min_samples_split=2):
    if len(np.unique(y)) == 1 or depth >= max_depth or len(y) < min_samples_split:
        values, counts = np.unique(y, return_counts=True)
        return Node(value=values[np.argmax(counts)])
    feature, threshold = best_split(X, y)
    if feature is None:
        values, counts = np.unique(y, return_counts=True)
        return Node(value=values[np.argmax(counts)])
    left_mask = X[feature] <= threshold
    right_mask = X[feature] > threshold
    left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_samples_split)
    right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, min_samples_split)
    return Node(feature, threshold, left, right)

def predict_tree(x, node):
    while node.value is None:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

def cart_predict(X, tree):
    return np.array([predict_tree(row, tree) for _, row in X.iterrows()])

def main():
    print("CART (Classification Tree)")
    csv_path = input("Enter path to your CSV data file: ")
    df = pd.read_csv(csv_path)
    print("Columns in your data:", list(df.columns))
    y_col = input("Enter the name of the target column: ")
    X = df.drop(columns=[y_col])
    y = df[y_col]
    max_depth = int(input("Enter max depth (ex 5): ") or 5)
    min_samples_split = int(input("Enter min samples split (ex 2): ") or 2)

    print("="*40)
    # Non-sklearn CART
    print("Using non-sklearn CART implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split)
    preds = cart_predict(X_test, tree)
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
        tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split)
        preds = cart_predict(X_test, tree)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")
    
    
    print("="*40)
    # Sklearn CART
    print("Using sklearn DecisionTreeClassifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    precision = precision_score(y_test.values, preds)
    print(f"Precision using hold-out validation: {precision * 100:.2f}%")

    precisions = np.zeros(NSPLIT)
    skf = StratifiedKFold(n_splits=NSPLIT, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = Scaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

if __name__ == "__main__":
    main()

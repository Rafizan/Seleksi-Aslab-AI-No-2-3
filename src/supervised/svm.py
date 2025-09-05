import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from scaler import Scaler, precision_score

def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)

def svm_train(X, y, lr=0.01, epochs=1000, C=1.0):
    X = np.c_[np.ones(X.shape[0]), X]
    y = y.values if isinstance(y, pd.Series) else y
    y = np.where(y == y.max(), 1, -1)
    weights = np.zeros(X.shape[1])
    for epoch in range(epochs):
        margin = y * np.dot(X, weights)
        mask = margin < 1
        grad = weights.copy()
        grad[1:] += -C * np.dot((y[mask]) , X[mask, 1:]) / X.shape[0]
        grad[0] += -C * np.sum(y[mask]) / X.shape[0]
        weights -= lr * grad
    return weights

def svm_predict(X, weights):
    X = np.c_[np.ones(X.shape[0]), X]
    preds = np.dot(X, weights)
    return (preds >= 0).astype(int)

def main():
    print("Linear SVM (Gradient Descent)")
    csv_path = input("Enter path to your CSV data file: ")
    df = pd.read_csv(csv_path)
    print("Columns in your data:", list(df.columns))
    y_col = input("Enter the name of the target column: ")
    X = df.drop(columns=[y_col])
    y = df[y_col]
    if len(np.unique(y)) > 2:
        print("Only binary classification is supported.")
        return
    
    lr = float(input("Enter learning rate (ex 0.01): ") or 0.01)
    epochs = int(input("Enter number of epochs (ex 1000): ") or 1000)
    C = float(input("Enter regularization parameter C (ex 1.0): ") or 1.0)

    print("="*40)
    # Non-sklearn SVM
    print("Using non-sklearn SVM implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    weights = svm_train(X_test, y_test, lr=lr, epochs=epochs, C=C)
    preds = svm_predict(X_test, weights)
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
        weights = svm_train(X_test, y_test, lr=lr, epochs=epochs, C=C)
        preds = svm_predict(X_test, weights)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

    print("="*40)
    # Sklearn SVM
    print("Using sklearn SVM implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = SVC(kernel='linear', C=C, max_iter=epochs)
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
        model = SVC(kernel='linear', C=C, max_iter=epochs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

if __name__ == "__main__":
    main()

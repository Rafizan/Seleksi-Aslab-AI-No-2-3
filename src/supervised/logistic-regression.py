import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from scaler import Scaler, precision_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_train(X_train, y_train, learning_rate=0.01, iterations=1000, regularization=None, lambda_reg=0.01):
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    weights = np.zeros(X_train.shape[1])
    bias = 0
    for i in range(iterations):
        z = np.dot(X_train, weights) + bias
        h = sigmoid(z)
        gradient_w = np.dot(X_train.T, (h - y_train)) / y_train.size
        gradient_b = np.sum(h - y_train) / y_train.size
        if regularization == "l2":
            gradient_w += (lambda_reg / y_train.size) * weights
        elif regularization == "l1":
            gradient_w += (lambda_reg / y_train.size) * np.sign(weights)
        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b
    return (weights, bias)

def logistic_regression_predict(X, weights, bias, threshold=0.5):
    X = np.c_[np.ones(X.shape[0]), X]
    probs = sigmoid(np.dot(X, weights) + bias)
    return (probs >= threshold).astype(int)

def main():
    print("Logistic Regression")

    csv_path = input("Enter path to your CSV data file: ")
    df = pd.read_csv(csv_path)
    print("Columns in your data:", list(df.columns))
    y_col = input("Enter the name of the target column: ")
    X = df.drop(columns=[y_col])
    y = df[y_col]

    lr = float(input("Enter learning rate (ex 0.01): ") or 0.01)
    iterations = int(input("Enter number of iterations (ex 1000): ") or 1000)
    regularization = input("Enter regularization type (none/l1/l2, default none): ") or "none"
    if regularization not in ["none", "l1", "l2"]:
        print("Invalid regularization type. Using 'none'.")
        regularization = None
    if regularization == "none":
        regularization = None

    print("="*40)
    # Non-sklearn Logistic Regression
    print("Using non-sklearn Logistic Regression implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    weights, bias = logistic_regression_train(X_train, y_train, learning_rate=lr, iterations=iterations, regularization=regularization)
    preds = logistic_regression_predict(X_test, weights, bias)
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
        weights, bias = logistic_regression_train(X_train, y_train, learning_rate=lr, iterations=iterations, regularization=regularization)
        preds = logistic_regression_predict(X_test, weights, bias)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

    print("="*40)
    # Sklearn Logistic Regression
    print("Using sklearn Logistic Regression implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(
        penalty=regularization,
        solver='lbfgs' if regularization in [None, 'l2', 'none'] else 'liblinear',
        max_iter=iterations,
        C=1/lr if regularization in ['l1', 'l2'] else 1e12
    )
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
        model = LogisticRegression(
            penalty=regularization,
            solver='lbfgs' if regularization in [None, 'l2', 'none'] else 'liblinear',
            max_iter=iterations,
            C=1/lr if regularization in ['l1', 'l2'] else 1e12
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

if __name__ == "__main__":
    main()

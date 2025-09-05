import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from scaler import Scaler, precision_score

def gaussian_nb_train(X_train, y_train):
    classes = np.unique(y_train)
    params = {}
    for c in classes:
        X_c = X_train[y_train == c]
        params[c] = {
            'mean': X_c.mean(axis=0).values,
            'var': X_c.var(axis=0).values,
            'prior': len(X_c) / len(X_train)
        }
    return params

def gaussian_nb_predict(X, params):
    classes = list(params.keys())
    n_samples = X.shape[0]
    n_classes = len(classes)
    posteriors = np.zeros((n_samples, n_classes))
    for idx, c in enumerate(classes):
        mean = params[c]['mean']
        var = params[c]['var']
        prior = params[c]['prior']
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var)) \
                        -0.5 * np.sum(((X - mean) ** 2) / (var), axis=1)
        log_prior = np.log(prior)
        posteriors[:, idx] = log_prior + log_likelihood
    return np.array([classes[i] for i in np.argmax(posteriors, axis=1)])

def main():
    print("Gaussian Naive Bayes")
    csv_path = input("Enter path to your CSV data file: ")
    df = pd.read_csv(csv_path)
    print("Columns in your data:", list(df.columns))
    y_col = input("Enter the name of the target column: ")
    X = df.drop(columns=[y_col])
    y = df[y_col]

    print("="*40)
    # Non-sklearn Gaussian Naive Bayes
    print("Using non-sklearn Gaussian Naive Bayes implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    params = gaussian_nb_train(X_train, y_train)
    preds = gaussian_nb_predict(X_test.values, params)
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
        params = gaussian_nb_train(X_train, y_train)
        preds = gaussian_nb_predict(X_test.values, params)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

    print("="*40)
    # Sklearn Gaussian Naive Bayes
    print("Using sklearn Gaussian Naive Bayes implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = GaussianNB()
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
        model = GaussianNB()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

if __name__ == "__main__":
    main()

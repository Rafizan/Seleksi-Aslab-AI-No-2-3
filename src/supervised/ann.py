import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from scaler import Scaler, precision_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)
def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    return (x > 0).astype(float)
def linear(x):
    return x
def linear_deriv(x):
    return np.ones_like(x)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def mse_loss_deriv(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
def cross_entropy_loss(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
def cross_entropy_loss_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

def init_weights(shape, method):
    if method == 'zeros':
        return np.zeros(shape)
    elif method == 'ones':
        return np.ones(shape)
    elif method == 'xavier':
        return np.random.randn(*shape) * np.sqrt(1 / shape[0])
    elif method == 'he':
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])
    else:
        return np.random.randn(*shape) * 0.01

class ANN:
    def __init__(self, layer_sizes, activations, init_method='xavier', loss='mse', reg=None, lambda_reg=0.01):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.init_method = init_method
        self.loss = loss
        self.reg = reg
        self.lambda_reg = lambda_reg
        self.weights = []
        self.biases = []
        self._init_params()
    def _init_params(self):
        for i in range(len(self.layer_sizes) - 1):
            w = init_weights((self.layer_sizes[i], self.layer_sizes[i+1]), self.init_method)
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    def _get_activation(self, name):
        if name == 'sigmoid':
            return sigmoid, sigmoid_deriv
        elif name == 'relu':
            return relu, relu_deriv
        elif name == 'linear':
            return linear, linear_deriv
        elif name == 'softmax':
            return softmax, None
        else:
            raise ValueError(f"Unknown activation: {name}")
    def forward(self, X):
        a = X
        activations = [a]
        zs = []
        for i, (w, b, act_name) in enumerate(zip(self.weights, self.biases, self.activations)):
            z = np.dot(a, w) + b
            zs.append(z)
            act_func, _ = self._get_activation(act_name)
            a = act_func(z)
            activations.append(a)
        return activations, zs
    def backward(self, X, y, activations, zs):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        # Loss derivative
        if self.loss == 'mse':
            delta = mse_loss_deriv(y, activations[-1])
        elif self.loss == 'cross_entropy':
            delta = cross_entropy_loss_deriv(y, activations[-1])
        else:
            raise ValueError('Unknown loss function')
        for l in reversed(range(len(self.weights))):
            act_func, act_deriv = self._get_activation(self.activations[l])
            if l != len(self.weights) - 1 or self.activations[l] != 'softmax':
                delta = delta * act_deriv(zs[l])
            grads_w[l] = np.dot(activations[l].T, delta)
            grads_b[l] = np.sum(delta, axis=0, keepdims=True)
            if l != 0:
                delta = np.dot(delta, self.weights[l].T)
        # Regularization
        if self.reg == 'l2':
            grads_w = [gw + self.lambda_reg * w for gw, w in zip(grads_w, self.weights)]
        elif self.reg == 'l1':
            grads_w = [gw + self.lambda_reg * np.sign(w) for gw, w in zip(grads_w, self.weights)]
        return grads_w, grads_b
    def fit(self, X, y, lr=0.01, epochs=1000, batch_size=32, verbose=True):
        y_proc = y.values
        if self.activations[-1] == 'softmax':
            n_classes = self.layer_sizes[-1]
            y_onehot = np.zeros((y_proc.shape[0], n_classes))
            y_onehot[np.arange(y_proc.shape[0]), y_proc.astype(int)] = 1
            y_proc = y_onehot
        
        n = X.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_shuffled = X.iloc[perm]
            y_shuffled = y_proc[perm]

            for i in range(0, n, batch_size):
                X_batch = X_shuffled.iloc[i:i+batch_size].values
                y_batch = y_shuffled[i:i+batch_size]
                
                activations, zs = self.forward(X_batch)
                grads_w, grads_b = self.backward(X_batch, y_batch, activations, zs)
                
                for l in range(len(self.weights)):
                    self.weights[l] -= lr * grads_w[l]
                    self.biases[l] -= lr * grads_b[l]
            
            if verbose and (epoch+1) % max(1, epochs//10) == 0:
                activations, _ = self.forward(X.values)
                if self.loss == 'mse':
                    y_loss_calc = y_proc if self.activations[-1] == 'softmax' else y_proc.reshape(-1, 1)
                    loss = mse_loss(y_loss_calc, activations[-1])
                else:
                    loss = cross_entropy_loss(y_proc, activations[-1])

    def predict(self, X):
        activations, _ = self.forward(X)
        if self.activations[-1] == 'softmax':
            return np.argmax(activations[-1], axis=1)
        else:
            return (activations[-1] >= 0.5).astype(int).flatten()

def main():
    print("Artificial Neural Network (ANN)")
    csv_path = input("Enter path to your CSV data file: ")
    df = pd.read_csv(csv_path)
    print("Columns in your data:", list(df.columns))
    y_col = input("Enter the name of the target column: ")
    X = df.drop(columns=[y_col])
    y = df[y_col]

    output_activation = input("Enter output activation (sigmoid/softmax/linear): ")
    n_layers = int(input("Enter number of hidden layers: "))
    neurons = [int(input(f"Enter number of neurons for layer {i+1}: ")) for i in range(n_layers)]
    input_size = X.shape[1] 
    n_classes = y.nunique()

    # For binary classification (sigmoid), output is 1 neuron. For multi-class (softmax), it's n_classes.
    output_size = 1 if n_classes == 2 and output_activation == 'sigmoid' else n_classes
    layer_sizes = [input_size] + neurons + [output_size]
    activations = []
    for i in range(n_layers):
        act = input(f"Enter activation for hidden layer {i+1} (sigmoid/relu/linear): ")
        activations.append(act)
    activations.append(output_activation)
    init_method = input("Enter initialization method (zeros/ones/xavier/he): ") or 'xavier'
    loss = input("Enter loss function (mse/cross_entropy): ") or 'mse'
    reg = input("Enter regularization (none/l1/l2): ") or None
    if reg == 'none':
        reg = None
    lambda_reg = float(input("Enter lambda for regularization (ex 0.01): ") or 0.01)
    lr = float(input("Enter learning rate (ex 0.01): ") or 0.01)
    epochs = int(input("Enter number of epochs (ex 1000): ") or 1000)
    batch_size = int(input("Enter batch size (ex 32): ") or 32)

    print("="*40)
    # Non-sklearn ANN
    print("Using non-sklearn ANN implementation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ann = ANN(layer_sizes, activations, init_method=init_method, loss=loss, reg=reg, lambda_reg=lambda_reg)
    ann.fit(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size)
    preds = ann.predict(X_test)
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
        ann = ANN(layer_sizes, activations, init_method=init_method, loss=loss, reg=reg, lambda_reg=lambda_reg)
        ann.fit(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size)
        preds = ann.predict(X_test)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

    print("="*40)
    # Sklearn ANN
    print("Using sklearn MLPClassifier implementation...")
    
    sklearn_activation = activations[0]
    if sklearn_activation == 'sigmoid':
        sklearn_activation = 'logistic'
    elif sklearn_activation == 'linear':
        sklearn_activation = 'identity'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=tuple(neurons),
                          activation=sklearn_activation,
                          solver='sgd',
                          learning_rate_init=lr,
                          max_iter=epochs,
                          batch_size=batch_size,
                          alpha=lambda_reg if reg == 'l2' else 0.0, # MLPClassifier only supports L2 reg (alpha)
                          random_state=42)
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
        model = MLPClassifier(hidden_layer_sizes=tuple(neurons),
                              activation=sklearn_activation,
                              solver='sgd',
                              learning_rate_init=lr,
                              max_iter=epochs,
                              batch_size=batch_size,
                              alpha=lambda_reg if reg == 'l2' else 0.0,
                              random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        precisions[fold] = precision_score(y_test.values, preds)
    print(f"Precision using {NSPLIT}-fold cross-validation: {np.mean(precisions) * 100:.2f}%")

if __name__ == "__main__":
    main()

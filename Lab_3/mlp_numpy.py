"""
MLP "з нуля" на NumPy - навчальний приклад (без TensorFlow).
Реалізує forward / backprop на маленькій задачі (XOR-подібна).
Це демонструє розуміння того, як працює нейронна мережа всередині.
"""
import numpy as np


class MLP:
    def __init__(self, layer_sizes, lr=0.1):
        self.lr = lr
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(
                2.0 / layer_sizes[i]
            )
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(out):
        return out * (1 - out)

    def forward(self, x):
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            z = activations[-1] @ w + b
            activations.append(self.sigmoid(z))
        return activations

    def backward(self, activations, y):
        # похибка вихідного шару
        delta = (activations[-1] - y) * self.sigmoid_derivative(activations[-1])
        grads_w = [activations[-2].T @ delta]
        grads_b = [np.sum(delta, axis=0, keepdims=True)]

        for i in range(len(self.weights) - 1, 0, -1):
            delta = (delta @ self.weights[i].T) * self.sigmoid_derivative(
                activations[i]
            )
            grads_w.insert(0, activations[i - 1].T @ delta)
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))

        return grads_w, grads_b

    def fit(self, X, y, epochs=10000):
        for epoch in range(epochs):
            activations = self.forward(X)
            grads_w, grads_b = self.backward(activations, y)
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]
            if epoch % 2000 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print("Епоха %d, MSE = %.6f" % (epoch, loss))

    def predict(self, X):
        return self.forward(X)[-1]


def main():
    # XOR-подібна задача: 2 вхідні, 4 приховані, 1 вихід
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    mlp = MLP([2, 4, 1], lr=0.5)
    mlp.fit(X, y, epochs=20000)

    preds = mlp.predict(X)
    print("\nПрогнози XOR:")
    for x_row, p in zip(X, preds):
        print("%s -> %.4f (клас: %d)" % (x_row, p[0], int(round(p[0]))))


if __name__ == "__main__":
    main()

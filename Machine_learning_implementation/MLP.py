import random as rnd
import numpy as np

class BaseMLP:
    def __init__(self, n_inputs, n_hidden, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        self.W1 = [
            [rnd.uniform(-1, 1) for _ in range(n_inputs)]
            for _ in range(n_hidden)
        ]

        self.B1 = [
            rnd.uniform(-1, 1)
            for _ in range(n_hidden)
        ]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def linear_sum(self, values, weights, bias):
        result = bias

        for v, w in zip(values, weights):
            result += v * w

        return result

    def forward_hidden_layer(self, x):
        z_hidden = []
        a_hidden = []

        for weights, bias in zip(self.W1, self.B1):
            z = self.linear_sum(x, weights, bias)
            a = self.sigmoid(z)

            z_hidden.append(z)
            a_hidden.append(a)

        return z_hidden, a_hidden

    def update_hidden_layer(self, hidden_deltas):
        for j in range(self.n_hidden):
            for i in range(self.n_inputs):
                gradient = hidden_deltas[j] * self.last_x[i]
                self.W1[j][i] -= self.learning_rate * gradient

            self.B1[j] -= self.learning_rate * hidden_deltas[j]

    def predict_batch(self, X):
        return [self.predict(x) for x in X]

class MLPRegressor(BaseMLP):
    def __init__(self, n_inputs, n_hidden, learning_rate=0.01):
        super().__init__(n_inputs, n_hidden, learning_rate)

        self.W2 = [
            rnd.uniform(-1, 1)
            for _ in range(n_hidden)
        ]

        self.B2 = rnd.uniform(-1, 1)

    def forward(self, x):
        self.last_x = x
        self.last_z_hidden, self.last_a_hidden = self.forward_hidden_layer(x)

        self.last_y_pred = self.linear_sum(
            self.last_a_hidden,
            self.W2,
            self.B2
        )

        return self.last_y_pred

    def calculate_output_delta(self, y_true):
        return self.last_y_pred - y_true

    def calculate_hidden_deltas(self, output_delta):
        hidden_deltas = []

        for j in range(self.n_hidden):
            delta = (
                self.W2[j]
                * output_delta
                * self.sigmoid_derivative(self.last_z_hidden[j])
            )

            hidden_deltas.append(delta)

        return hidden_deltas

    def update_output_layer(self, output_delta):
        for j in range(self.n_hidden):
            gradient = output_delta * self.last_a_hidden[j]
            self.W2[j] -= self.learning_rate * gradient

        self.B2 -= self.learning_rate * output_delta

    def backward(self, y_true):
        output_delta = self.calculate_output_delta(y_true)
        hidden_deltas = self.calculate_hidden_deltas(output_delta)

        self.update_output_layer(output_delta)
        self.update_hidden_layer(hidden_deltas)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            total_loss = 0

            for x_sample, y_true in zip(X, y):
                y_pred = self.forward(x_sample)

                loss = 0.5 * (y_pred - y_true) ** 2
                total_loss += loss

                self.backward(y_true)

            loss_mean = total_loss / len(X)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss_mean:.6f}")

    def predict(self, x):
        return self.forward(x)
    
class MLPClassifier(BaseMLP):
    def __init__(self, n_inputs, n_hidden, n_classes, learning_rate=0.01):
        super().__init__(n_inputs, n_hidden, learning_rate)

        self.n_classes = n_classes

        self.W2 = [
            [rnd.uniform(-1, 1) for _ in range(n_hidden)]
            for _ in range(n_classes)
        ]

        self.B2 = [
            rnd.uniform(-1, 1)
            for _ in range(n_classes)
        ]

    def softmax(self, values):
        max_value = max(values)

        exp_values = [
            np.exp(v - max_value)
            for v in values
        ]

        total = sum(exp_values)

        return [
            v / total
            for v in exp_values
        ]

    def one_hot(self, class_label):
        encoded = [0 for _ in range(self.n_classes)]

        # classes esperadas: 1, 2, 3, 4
        encoded[class_label - 1] = 1

        return encoded

    def forward(self, x):
        self.last_x = x
        self.last_z_hidden, self.last_a_hidden = self.forward_hidden_layer(x)

        self.last_z_output = []

        for k in range(self.n_classes):
            z = self.linear_sum(
                self.last_a_hidden,
                self.W2[k],
                self.B2[k]
            )

            self.last_z_output.append(z)

        self.last_y_pred = self.softmax(self.last_z_output)

        return self.last_y_pred

    def calculate_output_deltas(self, y_true):
        y_encoded = self.one_hot(y_true)

        output_deltas = []

        for pred, true in zip(self.last_y_pred, y_encoded):
            output_deltas.append(pred - true)

        return output_deltas

    def calculate_hidden_deltas(self, output_deltas):
        hidden_deltas = []

        for j in range(self.n_hidden):
            error_sum = 0

            for k in range(self.n_classes):
                error_sum += self.W2[k][j] * output_deltas[k]

            delta = error_sum * self.sigmoid_derivative(self.last_z_hidden[j])
            hidden_deltas.append(delta)

        return hidden_deltas

    def update_output_layer(self, output_deltas):
        for k in range(self.n_classes):
            for j in range(self.n_hidden):
                gradient = output_deltas[k] * self.last_a_hidden[j]
                self.W2[k][j] -= self.learning_rate * gradient

            self.B2[k] -= self.learning_rate * output_deltas[k]

    def cross_entropy_loss(self, y_true):
        y_encoded = self.one_hot(y_true)

        loss = 0

        for true, pred in zip(y_encoded, self.last_y_pred):
            if true == 1:
                loss -= np.log(pred + 1e-15)

        return loss

    def backward(self, y_true):
        output_deltas = self.calculate_output_deltas(y_true)
        hidden_deltas = self.calculate_hidden_deltas(output_deltas)

        self.update_output_layer(output_deltas)
        self.update_hidden_layer(hidden_deltas)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            total_loss = 0

            for x_sample, y_true in zip(X, y):
                self.forward(x_sample)

                loss = self.cross_entropy_loss(y_true)
                total_loss += loss

                self.backward(y_true)

            loss_mean = total_loss / len(X)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss_mean:.6f}")

    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x):
        probabilities = self.forward(x)

        max_index = probabilities.index(max(probabilities))

        return max_index + 1
# # mlp_predictor.py
# import numpy as np

# class SimpleMLP:
#     def __init__(self):
#         self.W1 = np.random.randn(1, 8)
#         self.b1 = np.zeros((1, 8))
#         self.W2 = np.random.randn(8, 3)
#         self.b2 = np.zeros((1, 3))
#         self.lr = 0.01

#     def relu(self, x):
#         return np.maximum(0, x)

#     def softmax(self, x):
#         exps = np.exp(x - np.max(x, axis=1, keepdims=True))
#         return exps / np.sum(exps, axis=1, keepdims=True)

#     def train(self, X, y, epochs=200):
#         y_oh = np.eye(3)[y]
#         for _ in range(epochs):
#             z1 = X @ self.W1 + self.b1
#             a1 = self.relu(z1)
#             z2 = a1 @ self.W2 + self.b2
#             probs = self.softmax(z2)
#             dz2 = probs - y_oh
#             dW2 = a1.T @ dz2
#             db2 = np.sum(dz2, axis=0, keepdims=True)
#             da1 = dz2 @ self.W2.T
#             dz1 = da1 * (z1 > 0)
#             dW1 = X.T @ dz1
#             db1 = np.sum(dz1, axis=0, keepdims=True)
#             self.W1 -= self.lr * dW1
#             self.b1 -= self.lr * db1
#             self.W2 -= self.lr * dW2
#             self.b2 -= self.lr * db2

#     def predict(self, freq):
#         X = np.array([[freq]])
#         a1 = self.relu(X @ self.W1 + self.b1)
#         probs = self.softmax(a1 @ self.W2 + self.b2)
#         return np.argmax(probs, axis=1)[0]

# # Train the model on dummy data
# X_train = np.array([[0.5], [0.8], [1.2], [1.5], [2.5], [2.8]])
# y_train = np.array([0, 0, 1, 1, 2, 2])
# mlp = SimpleMLP()
# mlp.train(X_train, y_train, epochs=200)
# def classify_frequency(freq):
#     return mlp.predict(freq)

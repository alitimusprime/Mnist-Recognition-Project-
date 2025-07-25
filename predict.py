import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load MNIST dataset
(X, y), _ = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model = tf.keras.models.load_model("E:\Ali work\mnist\mnist_model.keras")

# Predict
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))

accuracy = accuracy_score(y_test, predicted_classes)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load the full MNIST dataset (only training part)
(x_full, y_full), (_, _) = tf.keras.datasets.mnist.load_data()

# 2. Normalize pixel values (0 to 1) and add channel dimension
x_full = x_full.astype("float32") / 255.0
x_full = np.expand_dims(x_full, -1)  # Shape becomes (num_samples, 28, 28, 1)

# 3. Split manually: 80% training, 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    x_full, y_full, test_size=0.2, random_state=42
)

# 4. Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 digit classes (0-9)
])

# 5. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 7. Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# 8. Save the trained model
model.save("mnist_model.keras")
print("Model saved as mnist_model.keras")

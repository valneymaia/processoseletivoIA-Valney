import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_H5_PATH = BASE_DIR / "model.h5"


print("Carregando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Dados de treinamento: {x_train.shape}")
print(f"Dados de teste: {x_test.shape}")


print("\nConstruindo modelo CNN...")
model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        # Primeira camada convolucional
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Segunda camada convolucional
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Terceira camada convolucional
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        
        # Camadas densas
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

print("Resumo do modelo:")
model.summary()


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


print("\nTreinando modelo...")
batch_size = 128
epochs = 5

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=1,
)


print("\nAvaliando modelo nos dados de teste...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
test_probabilities = model.predict(x_test, verbose=0)
test_predictions = np.argmax(test_probabilities, axis=1)

f1_scores = []
for class_id in range(10):
    true_positive = np.sum((test_predictions == class_id) & (y_test == class_id))
    false_positive = np.sum((test_predictions == class_id) & (y_test != class_id))
    false_negative = np.sum((test_predictions != class_id) & (y_test == class_id))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    f1_scores.append(f1)

f1_macro = float(np.mean(f1_scores))

print(f"Loss final no teste: {test_loss:.4f}")
print(f"Acurácia final no teste: {test_accuracy * 100:.2f}%")
print(f"F1-score macro no teste: {f1_macro:.4f}")


print("\nSalvando modelo treinado como model.h5...")
model.save(MODEL_H5_PATH)
print(f"Modelo salvo com sucesso em: {MODEL_H5_PATH}")
print(f"Arquivo existe? {MODEL_H5_PATH.exists()}")

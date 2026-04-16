import tensorflow as tf
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_H5_PATH = BASE_DIR / "model.h5"
MODEL_TFLITE_PATH = BASE_DIR / "model.tflite"

print("Carregando modelo treinado...")
model = tf.keras.models.load_model(MODEL_H5_PATH)
print("Modelo carregado com sucesso!")

# Cria conversor para TensorFlow Lite
print("\nConvertendo modelo para TensorFlow Lite com Dynamic Range Quantization...")

# Cria o conversor com otimização por Dynamic Range Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

print("Salvando modelo otimizado como model.tflite...")
with open(MODEL_TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Modelo otimizado salvo com sucesso em: {MODEL_TFLITE_PATH}")
print(f"Arquivo existe? {MODEL_TFLITE_PATH.exists()}")

# Exibe informações sobre o tamanho
original_size = os.path.getsize(MODEL_H5_PATH) / (1024 * 1024)  # MB
optimized_size = os.path.getsize(MODEL_TFLITE_PATH) / (1024 * 1024)  # MB
reduction = (1 - (optimized_size / original_size)) * 100

print(f"\nResumo da Otimização:")
print(f"Tamanho original (model.h5): {original_size:.2f} MB")
print(f"Tamanho otimizado (model.tflite): {optimized_size:.2f} MB")
print(f"Redução de tamanho: {reduction:.2f}%")

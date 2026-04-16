import tensorflow as tf
import os

print("Carregando modelo treinado...")
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado com sucesso!")

# Cria conversor para TensorFlow Lite
print("\nConvertendo modelo para TensorFlow Lite com Dynamic Range Quantization...")

# Cria o conversor com otimização por Dynamic Range Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

print("Salvando modelo otimizado como model.tflite...")
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo otimizado salvo com sucesso!")

# Exibe informações sobre o tamanho
original_size = os.path.getsize("model.h5") / (1024 * 1024)  # MB
optimized_size = os.path.getsize("model.tflite") / (1024 * 1024)  # MB
reduction = (1 - (optimized_size / original_size)) * 100

print(f"\nResumo da Otimização:")
print(f"Tamanho original (model.h5): {original_size:.2f} MB")
print(f"Tamanho otimizado (model.tflite): {optimized_size:.2f} MB")
print(f"Redução de tamanho: {reduction:.2f}%")

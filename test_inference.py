import numpy as np
import tensorflow as tf
import os


def main() -> None:
    print("Carregando modelo TFLite...")
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("Carregando MNIST...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    index = np.random.randint(0, len(x_test))
    image_raw = x_test[index].astype("uint8")
    image = image_raw.astype("float32") / 255.0
    image = np.expand_dims(image, axis=(0, -1))

    image_path = "inference_sample.png"
    image_bytes = tf.io.encode_png(np.expand_dims(image_raw, axis=-1)).numpy()
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    if input_details["dtype"] != np.float32:
        image = image.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], image)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details["index"])[0]
    predicted_class = int(np.argmax(prediction))
    true_class = int(y_test[index])

    print(f"Amostra escolhida: {index}")
    print(f"Classe real: {true_class}")
    print(f"Classe prevista: {predicted_class}")
    print(f"Confiança: {prediction[predicted_class] * 100:.2f}%")
    print(f"Imagem salva em: {os.path.abspath(image_path)}")

    if os.name == "nt":
        try:
            os.startfile(image_path)
            print("Imagem aberta no visualizador padrão do Windows.")
        except OSError:
            print("Não foi possível abrir a imagem automaticamente.")


if __name__ == "__main__":
    main()
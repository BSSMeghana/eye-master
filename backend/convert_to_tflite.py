import tensorflow as tf

keras_model_path = "resnet50_retina_classifier.h5"

model = tf.keras.models.load_model(keras_model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quant_model = converter.convert()

quant_model_path = "resnet50_retina_classifier_quant.tflite"
with open(quant_model_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"Quantized model saved as {quant_model_path}")

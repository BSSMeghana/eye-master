import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('densenet.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('densenet.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Conversion complete: densenet.tflite saved.")

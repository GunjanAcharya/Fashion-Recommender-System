import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available and ready to use.")
    print("List of GPUs:", gpus)
else:
    print("No GPU detected. TensorFlow will use CPU.")
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(len(gpus))
for gpu in gpus:
    print(gpu)
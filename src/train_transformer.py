from datetime import datetime
from preprocess import Preprocessor
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.metrics import Accuracy, Precision, Recall
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from message_encoder import *
from keras.optimizers import Adam, RMSprop, SGD, Nadam

save_weights = True

# hyper parameters
# preprocessing
log_files = [i for i in range(745, 760)]            # list of ints representing the numbers of log files to use
logs_per_class = 100                                # How many datapoints per class should be collected if available
window_size = 20                                    # how many log messages to be considered in a single data point from sliding window
encoding_output_size = 16                           # size to be passed to the message_encoder, note that this is not neccessairily the shape of the output
message_encoder = BERTEncoder(encoding_output_size) # the message_encoder to be used. Can be TextVectorizationEncoder (uses keras.layers.TextVectorizer), BERTEncoder (only uses the BERT tokenizer) or BERTEmbeddingEncoder (also uses the BERT model)
test_ratio = 0.2                                    # percantage of the collected data that should be used for testing rather than training
extended_datetime_features = False                  # bool, whether the preprocessing should use a multitude of normalized features extracted from the date 
# lstm architecture
lstm_layers = 1                                     # int, how many lstm layers to use
lstm_units_per_layer = 50                           # int, how many lstm units per layer to use
dropout = 0.0                                       # float, which dropout value to use, 0.0 is equivalent to not using any dropout
recurrent_dropout = 0.0                             # float, same as with regular dropout
# training
epochs = 1000                                       # number of iterations to train
batch_size = 32                                     # int, number of samples processed before updating the model weights.
early_stopping_monitor = "val_loss"                 # what value to monitor for early_stopping. can be 'loss', 'val_loss', 'accuracy', 'val_accuracy', 'precision', 'val_precision', 'recall', 'val_recall', 'f1_score', 'val_f1_score'
early_stopping_patience = 10                        # int, number of epochs to wait after no improvement, if this is greater than epochs, EarlyStopping will not apply
early_stopping_restore_best = True                  # bool, if true keeps the best weights, not the final ones.
validation_split = 0.1
learning_rate = 0.001                               # float to specify learning rate of the optimizer
optimizer = Adam(learning_rate=learning_rate)       # optimizer, can be one of Adam, RMSprop, SGD (can have momentum parameter), Nadam

# preprocessing
pp = Preprocessor(log_files, 
                  message_encoder, 
                  logs_per_class=logs_per_class, 
                  window_size=window_size, 
                  extended_datetime_features=extended_datetime_features, 
                  volatile=True)
train_data, test_data = pp.stratified_split(test_ratio=test_ratio)
X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# transformer architecture
from keras.models import Model
from keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Add, GlobalAveragePooling1D
import tensorflow as tf
import keras


# Positional Encoding Layer
class PositionalEncoding(keras.layers.Layer):
    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_encoding = tf.cast(positions[:, tf.newaxis], tf.float32)
        return x + tf.math.sin(pos_encoding)

input_shape = pp.get_shape()  # (window_size, encoding_output_size)
num_classes = len(set(y_train))

model_dim = 128
num_heads = 4
ff_dim = 256
# input layer
inputs = Input(shape=input_shape)
# set to a fixed dimension using a Dense Layer
x = Dense(model_dim)(inputs)
x = PositionalEncoding()(x)

# transformer block
attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=model_dim//num_heads)(x, x)
attn_output = Dropout(dropout)(attn_output)
x = Add()([x, attn_output])
x = LayerNormalization(epsilon=1e-6)(x)

ff_output = Dense(ff_dim, activation='relu')(x)
ff_output = Dense(model_dim)(ff_output)
ff_output = Dropout(dropout)(ff_output)
x = Add()([x, ff_output])
x = LayerNormalization(epsilon=1e-6)(x)

x = Dense(64, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x[:, 0])

model = Model(inputs=inputs, outputs=x)
model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])



# train the model
early_stopping = EarlyStopping(monitor=early_stopping_monitor, 
                               patience=early_stopping_patience, 
                               restore_best_weights=early_stopping_restore_best)
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

# optionally save the model weights
filename = f"lstm_{lstm_layers}x{lstm_units_per_layer}_drop{dropout}_rec{recurrent_dropout}_enc{message_encoder.__class__.__name__.lower()}_{logs_per_class}logs_win{window_size}_lr{learning_rate}_bs{batch_size}_ep{epochs}_{early_stopping_monitor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if save_weights: model.save(f"{filename}.h5")

# Summary of the model
model.summary()

# Evaluate the model on test data
results = model.evaluate(X_test, y_test, verbose=0)
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# Predictions
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Classification report and F1 score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score (weighted): {f1:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig(f"{filename}.png")

# Print comparison of a few predictions vs actual values
a = []
for i in range(80):
    a.append(y_pred[i] == y_test[i])
    print(predictions[i])

print(a.count(True))
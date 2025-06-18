from datetime import datetime
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Add, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.metrics import Accuracy, Precision, Recall
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import keras
from positional_encoding import PositionalEncoding

from preprocessor import Preprocessor
from message_encoder import *

# === General settings ===
save_weights = True

# === Preprocessing parameters ===
log_files = [i for i in range(745, 754)]            # list of ints representing the numbers of log files to use
logs_per_class = 100                                # int, how many datapoints per class should be collected if available
window_size = 20                                    # int, how many log messages to be considered in a single data point from sliding window
encoding_output_size = 16                           # int, dimensionality of each message's encoded vector representation
message_encoder = BERTEncoder(encoding_output_size) # encoder object, can be TextVectorizationEncoder, BERTEncoder, or BERTEmbeddingEncoder
split_ratios = (4, 1)                               # tuple of ints, ratio of data split for training and testing
extended_datetime_features = False                  # bool, whether to use additional normalized datetime features
preprocessor_file = "./data/preprocessors/preprocessor_9files_100lpc_20ws_BERTencx16.json"  # str (path), if not empty and exists, loads existing preprocessor file instead of creating one

# === Transformer Hyperparameters ===
model_dim = 128                                     # int, dimension of the transformer hidden layers; must be divisible by num_heads
num_heads = 4                                       # int, number of attention heads in multi-head attention; must divide model_dim evenly
ff_dim = 256                                        # int, dimension of the feed-forward layer inside each transformer block
num_transformer_blocks = 2                          # int, how many stacked transformer blocks to use; generally 1–6 for lightweight models
dropout_rate = 0.1                                  # float in range [0.0, 1.0], dropout rate for regularization; 0.0 disables dropout

# === Training hyperparameters ===
epochs = 1000                                       # int, number of iterations (epochs) to train the model
batch_size = 32                                     # int, number of samples per training batch
early_stopping_monitor = "val_loss"                 # str, metric to monitor for early stopping; options: 'loss', 'val_loss', 'accuracy', 'val_accuracy', etc.
early_stopping_patience = 10                        # int, number of epochs with no improvement after which training will be stopped
early_stopping_restore_best = True                  # bool, whether to restore model weights from the epoch with the best monitored value
validation_split = 0.1                              # float in range (0.0, 1.0), fraction of training data to reserve for validation
learning_rate = 0.001                               # float > 0.0, learning rate used by the optimizer
optimizer = Adam(learning_rate=learning_rate)       # optimizer instance, can be Adam, RMSprop, SGD (optionally with momentum), or Nadam



# === Load or preprocess data ===
if os.path.isfile(preprocessor_file):
    pp = Preprocessor.load(preprocessor_file)
else:
    pp = Preprocessor(log_files, message_encoder,
                      logs_per_class=logs_per_class,
                      window_size=window_size,
                      extended_datetime_features=extended_datetime_features,
                      volatile=True)
    pp.preprocess()
    path = f"./data/preprocessors/preprocessor_{len(pp.loaded_files)}files_"
    enc_type = "BERTenc" if isinstance(pp.message_encoder, BERTEncoder) else "BERTemb" if isinstance(pp.message_encoder, BERTEmbeddingEncoder) else "TextVec"
    path += f"{logs_per_class}lpc_{window_size}ws_{enc_type}x{encoding_output_size}"
    if extended_datetime_features:
        path += "_extdt"
    path += ".json"
    if not os.path.isfile(path):
        pp.save(path)

# === Prepare data ===
train, test = pp.data.stratified_split(split_ratios)
X_train, y_train = train
X_test, y_test = test
input_shape = pp.data.entry_shape
num_classes = len(set(y_train))

# === Transformer Block ===
def transformer_block(x, model_dim, num_heads, ff_dim, dropout_rate):
    # Multi-Head Attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=model_dim // num_heads)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-Forward Network
    ff_output = Dense(ff_dim, activation='relu')(x)
    ff_output = Dense(model_dim)(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x

# === Build Transformer Model ===
inputs = Input(shape=input_shape)
x = Dense(model_dim)(inputs)  # Project to model dimension
x = PositionalEncoding()(x)

# Apply multiple transformer blocks
for _ in range(num_transformer_blocks):
    x = transformer_block(x, model_dim, num_heads, ff_dim, dropout_rate)

x = GlobalAveragePooling1D()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Training ===
early_stopping = EarlyStopping(monitor=early_stopping_monitor,
                               patience=early_stopping_patience,
                               restore_best_weights=early_stopping_restore_best)

model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=validation_split,
          callbacks=[early_stopping])

# === Save model ===
filename = f"transformer_{num_transformer_blocks}x{model_dim}_drop{dropout_rate}_enc{message_encoder.__class__.__name__.lower()}_{logs_per_class}logs_win{window_size}_lr{learning_rate}_bs{batch_size}_ep{epochs}_{early_stopping_monitor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if save_weights: model.save(f"{filename}.keras")

# === Evaluate ===
model.summary()
results = model.evaluate(X_test, y_test, verbose=0)
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# === Predictions & Report ===
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score (weighted): {f1:.4f}")

# === Confusion Matrix ===
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(f"{filename}.png")
plt.show()

# === Preview Predictions ===
for i in range(min(80, len(y_pred))):
    print(f"True: {y_test[i]}, Pred: {y_pred[i]}, Prob: {predictions[i]}")

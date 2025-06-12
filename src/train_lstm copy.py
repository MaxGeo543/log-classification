from datetime import datetime
from pre import Preprocessor
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
import os

save_weights = True

# hyper parameters
# preprocessing
log_files = [i for i in range(745, 754)]            # list of ints representing the numbers of log files to use
logs_per_class = 100                                # How many datapoints per class should be collected if available
window_size = 20                                    # how many log messages to be considered in a single data point from sliding window
encoding_output_size = 16                           # size to be passed to the message_encoder, note that this is not neccessairily the shape of the output
message_encoder = BERTEncoder(encoding_output_size) # the message_encoder to be used. Can be TextVectorizationEncoder (uses keras.layers.TextVectorizer), BERTEncoder (only uses the BERT tokenizer) or BERTEmbeddingEncoder (also uses the BERT model)
split_ratios = (4, 1)                               # percantage of the collected data that should be used for testing rather than training
extended_datetime_features = False                  # bool, whether the preprocessing should use a multitude of normalized features extracted from the date 
preprocessor_file = "./data/preprocessors/preprocessor_9files__100lpc_20ws_BERTencx16.json"  #                             # if this is a string with content, load the file instead of creating a new preprocessor

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

if os.path.isfile(preprocessor_file):
    pp = Preprocessor.load(preprocessor_file)
else:
    # preprocessing
    pp = Preprocessor(log_files, 
                    message_encoder, 
                    logs_per_class=logs_per_class, 
                    window_size=window_size, 
                    extended_datetime_features=extended_datetime_features, 
                    volatile=True)
    pp.preprocess()

    # save the dataset if it doesn't exist already!
    path = f"./data/preprocessors/preprocessor_{len(pp.loaded_files)}files_"
    m = "BERTenc" if isinstance(pp.message_encoder, BERTEncoder) else "BERTemb" if isinstance(pp.message_encoder, BERTEmbeddingEncoder) else "TextVec" if isinstance(pp.message_encoder, TextVectorizationEncoder) else "enc"
    path += f"_{logs_per_class}lpc_{window_size}ws_{m}x{encoding_output_size}"
    if extended_datetime_features: path += "_extdt"
    path += ".json"
    if not os.path.isfile(path):
        pp.save(path)

train, test = pp.data.stratified_split((4, 1))
X_train, y_train = train
X_test, y_test = test

# lstm architecture
model = Sequential()
input_shape = pp.data.entry_shape
print(input_shape)
# First LSTM layer
if lstm_layers > 1:
    model.add(LSTM(lstm_units_per_layer, input_shape=input_shape, return_sequences=True))
else:
    model.add(LSTM(lstm_units_per_layer, input_shape=input_shape))  # single layer, no sequences returned
# Intermediate LSTM layers (if any)
for i in range(lstm_layers - 2):
    model.add(LSTM(lstm_units_per_layer, return_sequences=True))
# Last LSTM layer (no return_sequences, output fed into Dense)
if lstm_layers > 1:
    model.add(LSTM(lstm_units_per_layer))
# Dense Layer
num_classes = len(set(y_train))
model.add(Dense(num_classes, activation='softmax'))
# compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
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
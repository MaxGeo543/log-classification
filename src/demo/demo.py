from preprocess import Preprocessor
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense



# input shape
# X_train shape: (0.8*4*logs_per_class, window_size, 5)
# y_train shape: (0.8*4*logs_per_class,)
pp = Preprocessor([i for i in range(745, 760)], logs_per_class=100, window_size=20, volatile=True)
train_data, test_data = pp.stratified_split()
X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Build the model
model = Sequential()
model.add(LSTM(50, input_shape=(pp.window_size, 5)))  # 50 units in LSTM layer
num_classes = len(set(y_train))  # Assuming labels are integers
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Summary of the model
model.summary()
test_loss = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss}")



predictions = model.predict(X_test)

# Print a few predictions vs actuals
for i in range(5):
    print(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")
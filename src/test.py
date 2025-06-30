from positional_encoding import PositionalEncoding
from keras.models import load_model as _load_model
from keras.models import Sequential
import numpy as np
import random
import os
import tensorflow as tf
from preprocessor import Preprocessor



preprocessor_file = "./data/preprocessors/preprocessor_20_smallest_files_100lpc_20ws_BERTencx16.zip"
# model_weights = "./models/vary_lstm/lstm_1x50_drop0.0_rec0.0_lr0.001_bs32_ep1000_earlystpval_loss10True_20250624_145128.keras"
model_weights = "./models/vary_transformer_parameters/transformer_2x128_heads4_ffdim256_drop0.2_lr0.001_bs32_ep1000_earlystpval_loss10True_20250619_141304.keras"

def load_model(weights_path: str):
    model = _load_model(weights_path, custom_objects={'PositionalEncoding': PositionalEncoding})
    return model

def test_random_line(model: Sequential, preprocessor: Preprocessor, directory, seed = 42):
    
    # get a random file
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None, None  # or raise an exception
    
    file = os.path.join(directory, random.choice(files))

    # get a random file number
    line = 0
    with open(file, 'r') as f:
        line_count = sum(1 for _ in f)
    if line_count == 0:
        raise ValueError("File is empty.")
    line = random.randint(0, line_count - 1)

    # preprocess and annotate the event
    print(f"file {file} line {line}")
    vec, label, events, _ = preprocessor.preprocess_log_line(file, line)
    if vec is None:
        return None, None
    vec = np.expand_dims(vec, axis=0)

    predictions = model.predict(vec)
    pred = np.argmax(predictions, axis=1)
    actual = label

    # print(events)
    # print(vec)
    print(predictions)
    print(f"predicted: {pred}")
    print(f"actual: {actual}")
    print()

    return pred, actual


if __name__ == "__main__":
    if False:
        model = load_model(model_weights)
        preprocessor = Preprocessor.load(preprocessor_file)
        
        vec, label, events, _ = preprocessor.preprocess_log_line("./data/CCI/CCLog-backup.752.log", 48597)
        vec = np.expand_dims(vec, axis=0)
        for i in range(len(model.layers)):
            intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.layers[i].output)
            out = intermediate_model.predict(vec)
            print("LAYER", i)
            print(out)
        
        
        quit()
    elif True:
        model = load_model(model_weights)
        preprocessor = Preprocessor.load(preprocessor_file)
        
        vec, label, events, _ = preprocessor.preprocess_log_line("./data/CCI/CCLog-backup.752.log", 48597)
        vec = np.expand_dims(vec, axis=0)
        final_hidden = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        output = final_hidden.predict(vec)
        print(output)  # Check for large or weird values

        last_layer = model.layers[-1]
        print(last_layer(output)) 
        quit()
    

    model = load_model(model_weights)
    preprocessor = Preprocessor.load(preprocessor_file)
    data_dir = "./data/CCI"

    for i in range(10):
        prediction, actual = test_random_line(model, preprocessor, data_dir)
        # print(prediction, actual)
from positional_encoding import PositionalEncoding, PositionalEncoding2
from keras.models import load_model as _load_model
from keras.models import Sequential
import numpy as np
import random
import os
import tensorflow as tf
from preprocessor import Preprocessor

import json

preprocessor_file = "./data/preprocessors/preprocessor_20_smallest_files_100lpc_20ws_BERTencx16.zip"


def load_model(weights_path: str):
    model = _load_model(weights_path, custom_objects={'PositionalEncoding': PositionalEncoding, 'PositionalEncoding2': PositionalEncoding2})
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
    try:
        vec, label, events, _ = preprocessor.preprocess_log_line(file, line)
    except:
        print("Error: Label encoder got an unknown label")
        return "label", None
    
    if vec is None:
        print("Error: could't preprocess Datapoint, the line was likely too close to the end")
        print()
        return "datapoint", None
    vec = np.expand_dims(vec, axis=0)

    predictions = model.predict(vec)
    pred = np.argmax(predictions, axis=1)
    actual = label

    print(predictions)

    if tf.math.reduce_any(tf.math.is_nan(predictions)):
        print()
        return "nan", None

    print(f"predicted: {pred}")
    print(f"actual: {actual}")
    print()

    return pred, actual


if __name__ == "__main__":
    
    preprocessor = Preprocessor.load(preprocessor_file)
    
    model_results = {}

    directory = "./models/vary_lstm"
    model_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith("keras")]
    for model_weights in model_files:
        print(model_weights)
        model = load_model(os.path.join(directory, model_weights))
        model_results[model_weights] = {
            "nans": 0,
            "datapoint errors": 0,
            "label errors": 0,
            "pred_actual pairs": [],
            "correct predictions": 0
        }
        i = 0
        while i < 50:
            pred, actual = test_random_line(model, preprocessor, "./data/CCI")

            if pred == "datapoint":
                model_results[model_weights]["datapoint errors"] += 1
            elif pred == "label":
                model_results[model_weights]["label errors"] += 1
            elif pred == "nan":
                model_results[model_weights]["nans"] += 1
            else:
                if int(pred[0]) == actual: 
                    model_results[model_weights]["correct predictions"] += 1
                model_results[model_weights]["pred_actual pairs"].append((int(pred[0]), actual))
                i += 1
    print(model_results)

    with open("lstm_results.json", "w") as f:
        json.dump(model_results, f, indent=2)

    
from positional_encoding import PositionalEncoding, PositionalEncoding2
from keras.models import load_model as _load_model
from keras.models import Sequential
import numpy as np
import random
import os
import tensorflow as tf
from preprocessor import Preprocessor

from collections import defaultdict
import json
from eval_models import load_model, test_random_line


preprocessor_file = "./data/preprocessors/preprocessor_20_smallest_files_100lpc_20ws_BERTencx16.zip"
# model_weights = "./models/vary_lstm/lstm_1x50_drop0.0_rec0.0_lr0.001_bs32_ep1000_earlystpval_loss10True_20250624_145128.keras"
# this model does not create nans: model_weights = "./models/vary_transformer_parameters/transformer_2x128_heads4_ffdim256_drop0_lr0.001_bs32_ep1000_earlystpval_loss10True_20250619_135242.keras"
# this model does also not create nans and additionally has more class 0s: model_weights = "./models/vary_transformer_parameters/transformer_8x512_heads8_ffdim512_drop0.2_lr0.001_bs32_ep1000_earlystpval_loss10True_20250619_141410.keras"
# this model does also not create nans: model_weights = "./models/vary_transformer_parameters/transformer_3x128_heads4_ffdim256_drop0_lr0.001_bs32_ep1000_earlystpval_loss10True_20250619_140632.keras"


if __name__ == "__main__":
    preprocessor = Preprocessor.load(preprocessor_file)
    
    model_results = {}

    model_weights = "./models/transformer_2x128_heads4_ffdim256_drop0_lr0.001_bs32_ep1000_earlystpval_loss10True_20250703_063902.keras"

    print(model_weights)
    model = load_model(model_weights)
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

    
import numpy as np
import random
import os
import sys
import glob
import tensorflow as tf
from preprocessor import Preprocessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from eval_models import load_model, test_random_line

def get_files_with_extension(directory, extension):
    pattern = os.path.join(directory, f'*.{extension.lstrip(".")}')
    return glob.glob(pattern)

def get_files_with_extension_recursive(directory, extension):
    pattern = os.path.join(directory, '**', f'*.{extension.lstrip(".")}')
    return glob.glob(pattern, recursive=True)

# preprocessor_path = "../data/preprocessors/preprocessor_20_smallest_files_100lpc_20ws_BERTencx16.zip"
preprocessor_path = "./data/preprocessors/preprocessor_3_newest_files_100lpc_20ws_BERTencx16.zip"
pp = Preprocessor.load(preprocessor_path)

X, y = pp.data.as_xy_arrays()

model_dir = "./models/vary training parameters"
test_dir = "./models/vary training parameters"
# model_dir = "./models/vary_transformer_parameters"
# test_dir = "./models/vary_transformer_parameters/tests"

os.makedirs(test_dir, exist_ok=True)

models = get_files_with_extension_recursive(model_dir, "keras")
for i, model_path in enumerate(models):
    print(f"{i+1}/{len(models)}")
    model = load_model(model_path)
    m_name = os.path.splitext(os.path.basename(model_path))[0]


    # Get predictions
    y_pred_probs = model.predict(X)
    y_pred = y_pred_probs.argmax(axis=1)  # For softmax outputs

    # True labels
    y_true = y.argmax(axis=1) if y.ndim > 1 else y

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    f = open(f"{test_dir}/test_{m_name}-metrics.txt", "w")
    ostd = sys.stdout
    sys.stdout = f

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(f"{test_dir}/test_{m_name}-confusionmatrix.png")
    # plt.show()

    sys.stdout = ostd
    f.close()
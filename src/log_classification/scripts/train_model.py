from rich import print
import os
from typing import Any, Dict, List

from log_classification.util import get_sorted_log_numbers_by_size
from log_classification.dataset import Dataset
from log_classification.classes import classes, annotate as cl_annotate
from log_classification.lstm import LSTMClassifierLayer
from log_classification.classifier import Classifier
from log_classification.transformer import TransformerClassifierLayer
from log_classification.preprocessor import Preprocessor
from log_classification.data_augmentation import remove_and_pad
from log_classification.encoders.datetime_encoder import DatetimeEncoder
from log_classification.encoders.datetime_features import DatetimeFeature, DatetimeFeatureBase
from log_classification.encoders.loglevel_encoder import LogLevelEncoder, LogLevelOrdinalEncoder, LogLevelOneHotEncoder
from log_classification.encoders.message_encoder import MessageEncoder, MessageTextVectorizationEncoder, MessageBERTEmbeddingEncoder, MessageBERTEncoder
from log_classification.encoders.function_encoder import FunctionEncoder, FunctionLabelEncoder, FunctionOneHotEncoder, FunctionOrdinalEncoder
from log_classification.encoders.classes_encoder import ClassesEncoder, ClassesLabelBinarizer, ClassesLabelEncoder

PREPROCESSOR_PATH = DATASET_PATH = None

# large dataset (200 logs per class, 50 logs per window)
# PREPROCESSOR_PATH = r".\preprocessors\[chbt2s4V18eW9fsI][20250925_142809]preprocessor-window_size_50.json"
# DATASET_PATH = r".\data\datasets\[20250925_151312]dataset.npz"

# smaller dataset (100 logs per class, 20 logs per window)
PREPROCESSOR_PATH = r".\preprocessors\[1AwJTQMuN-CZAk6S][20250925_154810]preprocessor-window_size_20.json"
DATASET_PATH = r".\data\datasets\[20250925_161456]dataset.npz"

VERBOSE = True
LOGS_PER_CLASS = 200
EVALUATION_NAME = "evaluations/test"
MODEL_SAVE_NAME = "pp_small.keras"
MODEL_TYPE = "lstm" # or transformer

PREPROCESSOR_CONFIG = {
    "name": "preprocessor-window_size_20",
    "window_size":       20,
    "classes_encoder":   ClassesLabelEncoder(),
    "message_encoder":   MessageBERTEmbeddingEncoder(output_mode="cls"),
    "log_level_encoder": LogLevelOneHotEncoder(),
    "function_encoder":  FunctionOrdinalEncoder(),
    "datetime_encoder":  DatetimeEncoder([
                            DatetimeFeature.hour.cyclic.normalized
                         ])
}

LSTM_CONFIG = {
    "lstm_layers":         1,
    "units":              50,
    "dropout":           0.0,
    "recurrent_dropout": 0.0,
}

TRANSFORMER_CONFIG = {
    "transformer_layers": 2,
    "model_dim":          128,
    "num_heads":          4,
    "ff_dim":             256,
    "dropout":            0.1,
    "hidden_units":       64,
}

CLASSIFIER_CONFIG = {
    "data_split_ratios": (6, 1, 1, 0),
    "learning_rate":     0.001,
    "metrics":           ["accuracy"],
    "seed":              None,
}

TRAINING_CONFIG = {
    "epochs":                  1000, 
    "batch_size":              32,

    "es_monitor":              "val_loss",
    "es_patience":             200,
    "es_restore_best":         True,

    "pl_unlabeled_data":       None,
    "pl_interval":             10,
    "pl_confidence_threshold": 32,

    "data_augmentation_functions": [
        (0.2, lambda x, y, _1, _2: remove_and_pad(x, y))
    ]
}

LOGFILE_PATTERN = "./data/CCI/CCLog-backup.{num}.log"
LOGS_TO_LOAD = get_sorted_log_numbers_by_size("./data/CCI")[:40]


def get_preprocessor(src: str | Dict[str, Any],
                     logs_to_load: List[int],
                     verbose: bool = True,) -> Preprocessor:
    """
    Loads a preprocessor object from a source file or a dictionary defining preprocessor properties.
    
    If `src` is a string, it is expected to be the file path to a preprocessor file and the preprocessor will be loaded.

    If `src` is a dictionary, it will create a new preprocessor, load log files, initialize encoders and then save everything. This process may take a while. 

    :param src: The file path of the source file or the dictionary containing preprocessor properties
    :param logs_to_load: List of numbers of log files to load
    :param verbose: Whether to print information to the console
    :return: A `Preprocessor` object
    """
    
    if verbose: print("[cyan]Loading preprocessor...[/cyan]")
    
    if isinstance(src, str):
        if verbose: print(f"[cyan]Loading preprocessor from file: {src}[/cyan]")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Preprocessor file not found: {src}")

        pp = Preprocessor.load(src, annotate=cl_annotate, verbose=verbose)
        if verbose: print("[green]Preprocessor successfully loaded.[/green]")
    else:
        if verbose: print("[cyan]Creating new preprocessor from config...[/cyan]")

        config = src

        necessary_fields = {"window_size", "classes_encoder", "log_level_encoder", "datetime_encoder", "function_encoder", "message_encoder"}
        if any(arg not in config for arg in necessary_fields):
            raise ValueError("All encoder arguments must be provided in config. Missing fields: " + ", ".join(necessary_fields - config.keys()))

        pp = Preprocessor(
            name=config.get("name", "train_model_pp"),

            window_size=config["window_size"],
            classes=classes,

            message_encoder=config["message_encoder"],
            function_encoder=config["function_encoder"],
            datetime_encoder=config["datetime_encoder"],
            log_level_encoder=config["log_level_encoder"],
            classes_encoder=config["classes_encoder"],
            
            verbose=verbose
        )
        if verbose: print("[green]Preprocessor successfully created.[/green]")

        if verbose: print("[cyan]Load log files...[/cyan]")
        pp.load_logfiles([LOGFILE_PATTERN.format(num=num) for num in logs_to_load])
        if verbose: print("[green]Log files successfully loaded.[/green]")

        if verbose: print("[cyan]Initializing encoders...[/cyan]")
        pp.initialize_encoders()
        if verbose: print("[green]Encoders successfully initialized.[/green]")

        if verbose: print("[cyan]Saving preprocessor to file...[/cyan]")
        files = pp.save("./preprocessors")
        if verbose: 
            print(f"[green]Preprocessor successfully saved to:[/green]")
            for file in files:
                print(f"[green] - {file}[/green]")


    return pp

def get_dataset(src: str | Preprocessor,
                logs_to_load: List[int],
                logs_per_class: int = 100, 
                verbose: bool = True) -> Dataset:
    """
    Loads a dataset using either a source file or a Preprocessor with `logs_per_class`. 

    If `src` is a string, it is expected to be the file path to a dataset file and the dataset will be loaded. `logs_per_class` won't be used for this

    If `src` is a Preprocessor, the preprocessor is used to create a new dataset with `logs_per_class` logs per class. 
    If the preprocessor has no loaded files, files will be loaded, and the preprocessor freshly initialized. Then the dataset will be preprocessed and annotated. 
    Finally the datasets will be saved as files.

    :param src: The file path of the source file or the preprocessor object used to create new dataset
    :param logs_to_load: List of numbers of log files to load
    :param verbose: Whether to print information to the console
    :return: A `Dataset` object
    """
    
    if verbose: print("[cyan]Loading dataset...[/cyan]")
    # create dataset
    if isinstance(src, str):
        if verbose: print("[cyan]Loading dataset from file...[/cyan]")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Data file not found: {src}")
        dataset = Dataset.load(src)
        if verbose: print("[green]dataset successfully loaded.[/green]")
        return dataset
    elif isinstance(src, Preprocessor):
        pp = src
        if verbose: print("[cyan]Loading dataset by preprocessing and annotating logs...[/cyan]")
        if len(pp.events) == 0:
            if verbose: print("[cyan]Loading log files...[/cyan]")
            pp.load_logfiles([LOGFILE_PATTERN.format(num=num) for num in logs_to_load])
            if verbose: print(f"[green]Log files successfully loaded. ({len(pp.events) = })[/green]")
            if verbose: print("[cyan]Initializing encoders...[/cyan]")
            pp.initialize_encoders()
            if verbose: print("[green]Encoders successfully initialized.[/green]")

        if verbose: print("[cyan]Preprocessing dataset...[/cyan]")
        dataset = pp.preprocess_dataset(
            logs_per_class=logs_per_class,
            force_same_logs_per_class=False,
            max_events=None
        )
        if verbose: print("[green]Dataset successfully preprocessed.[/green]")
        
        if verbose: print("[cyan]Saving dataset to file...[/cyan]")
        ds_file = dataset.save("./data/datasets/[{timestamp}]dataset.npz", True)
        if verbose: print(f"[green]Dataset successfully saved to {ds_file}.[/green]")

        return dataset
    else:
        raise ValueError("Invalid source type.")

def get_model(model_type: str, 
              model_config: Dict[str, Any],
              classifier_config: Dict[str, Any],
              preprocessor: Preprocessor,
              dataset: Dataset) -> Classifier:
    """
    Get a `Classifier` that can be trained and used to classify log data. 

    :param model_type: The type of model to use in the classifier. Currently supports 'lstm' or 'transformer'
    :param model_config: The config and hyperparameters used to build the model. Specific to `model_type`
    :param classifier_config: The config and hyperparameters used to build the classifier, not specific to `model_type`
    :param preprocessor: A `Preprocessor` object
    :param dataset: A `Dataset` object
    :return: A `Dataset` object
    """
    model_type = model_type.lower()
    ClassifierLayer = LSTMClassifierLayer if model_type == "lstm" else TransformerClassifierLayer if model_type == "transformer" else None
    if ClassifierLayer is None:
        raise ValueError(f"Unknown model type: {model_type}")

    return Classifier(
        preprocessor=preprocessor,
        dataset=dataset,

        classifier_layer=ClassifierLayer(num_classes=len(classes.values), **model_config),

        **classifier_config
        )


if __name__ == "__main__":
    # get preprocessor
    preprocessor = get_preprocessor(
        PREPROCESSOR_PATH or PREPROCESSOR_CONFIG,
        LOGS_TO_LOAD,
        verbose=VERBOSE
        )

    # get dataset
    dataset = get_dataset(
        DATASET_PATH or preprocessor, 
        LOGS_TO_LOAD,
        verbose=VERBOSE, 
        logs_per_class=LOGS_PER_CLASS
        )

    # get model
    model = get_model(MODEL_TYPE, 
                      LSTM_CONFIG if MODEL_TYPE == "lstm" else TRANSFORMER_CONFIG,
                      CLASSIFIER_CONFIG,
                      preprocessor,
                      dataset)
    
    # train model
    model.train(**TRAINING_CONFIG)

    

    # evaluate model
    model.evaluate(EVALUATION_NAME)

    model.save(MODEL_SAVE_NAME)

    

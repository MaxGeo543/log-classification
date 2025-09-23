from preprocessor import Preprocessor, Dataset
from util import get_sorted_log_numbers_by_size
from encoders.datetime_encoder import DatetimeEncoder
from encoders.datetime_features import DatetimeFeature, DatetimeFeatureBase
from encoders.loglevel_encoder import *
from encoders.message_encoder import *
from encoders.function_encoder import *
from encoders.classes_encoder import *
from classes import classes, annotate as cl_annotate
from encoders.encoder_type import EncoderType
from util import *
import os
from rich import print

log_numbers = get_sorted_log_numbers_by_size("./data/CCI")[:20]
# log_numbers = [772, 771, 770]

load_preprocessor = False
pp_path = "./preprocessors/[Hv_-zDu9v7jr4Sam][20250811_105705]preprocessor_test.json"
pp_name = "preprocessor_test"
pp_window_size = 20
pp_classes = classes
annotation_func = cl_annotate
pp_classes_encoder = ClassesLabelBinarizer()
pp_log_level_encoder = LogLevelOrdinalEncoder()
pp_datetime_encoder = DatetimeEncoder([DatetimeFeature.second.since_midnight, DatetimeFeature.day.since_epoch])
pp_function_encoder = FunctionOneHotEncoder(min_frequency=3, max_categories=1000)
pp_message_encoder = MessageTextVectorizationEncoder(max_tokens=1000, output_mode="count")



num_logfiles = 1000  # Number of log files to use for preprocessing

load_dataset = False
dataset_path = "./data/datasets/dataset.npz" if load_preprocessor else "./data/datasets/[{timestamp}]dataset.npz"
ds_logs_per_class = 100
ds_shuffle = True
ds_max_events = None

print()


# preprocessing
if load_preprocessor:
    print("Loading preprocessor from:", pp_path)
    pp = Preprocessor.load(pp_path, annotate=annotation_func)
    print("[green]Preprocessor loaded successfully.[/green]")
else:
    print("Instantiate new preprocessor")
    pp = Preprocessor(
        message_encoder=pp_message_encoder,
        function_encoder=pp_function_encoder,
        datetime_encoder=pp_datetime_encoder,
        log_level_encoder=pp_log_level_encoder,
        classes_encoder=pp_classes_encoder,
        classes=pp_classes,
        window_size=pp_window_size,
        name=pp_name,
        verbose=True
    )

    log_file_paths = [
        rf".\data\CCI\CCLog-backup.{i}.log"
        for i in log_numbers
    ][:num_logfiles]
    total_size = sum(os.path.getsize(path) for path in log_file_paths)
    print(f"load {len(log_file_paths)} log files ({total_size / (1024 * 1024):.2f} MB).")
    pp.load_logfiles(log_file_paths)
    print("[green]Log files loaded successfully.[/green]")

    print("Initializing encoders...")
    pp.initialize_encoders()
    print("[green]Encoders initialized successfully.[/green]")

    # Encoder dimension validation
    print("[yellow]Validating encoder output dimensions...[/yellow]")

    # LogLevel Encoder
    sample_log_level = "Info"
    log_level_encoded = pp_log_level_encoder.encode(sample_log_level)
    log_level_dim = pp_log_level_encoder.get_dimension()
    assert (
        hasattr(log_level_encoded, "__len__") and len(log_level_encoded) == log_level_dim
    ) or (
        hasattr(log_level_encoded, "shape") and log_level_encoded.shape[-1] == log_level_dim
    ), f"LogLevelEncoder: output dimension {len(log_level_encoded)} != {log_level_dim}"

    # Datetime Encoder
    import datetime
    sample_datetime = datetime.datetime.now()
    datetime_encoded = pp_datetime_encoder.extract_date_time_features(sample_datetime)
    datetime_dim = pp_datetime_encoder.get_dimension()
    assert (
        hasattr(datetime_encoded, "__len__") and len(datetime_encoded) == datetime_dim
    ) or (
        hasattr(datetime_encoded, "shape") and datetime_encoded.shape[-1] == datetime_dim
    ), f"DatetimeEncoder: output dimension {len(datetime_encoded)} != {datetime_dim}"

    # Function Encoder
    sample_function = "C_line_Control_Server.WorkbenchTracking.SaveLogFiles"
    function_encoded = pp_function_encoder.encode(sample_function)
    function_dim = pp_function_encoder.get_dimension()
    assert (
        hasattr(function_encoded, "__len__") and len(function_encoded) == function_dim
    ) or (
        hasattr(function_encoded, "shape") and function_encoded.shape[-1] == function_dim
    ), f"FunctionEncoder: output dimension {len(function_encoded)} != {function_dim}"

    # Message Encoder
    sample_message = "This is a test log message."
    message_encoded = pp_message_encoder.encode(sample_message)
    message_dim = pp_message_encoder.get_dimension()
    assert (
        hasattr(message_encoded, "__len__") and len(message_encoded) == message_dim
    ) or (
        hasattr(message_encoded, "shape") and message_encoded.shape[-1] == message_dim
    ), f"MessageEncoder: output dimension {len(message_encoded)} != {message_dim}"

    print("[green]Encoder output dimensions validated successfully.[/green]")

    
    print("saving preprocessor...")
    files = pp.save("./preprocessors")
    print(f"[green]Preprocessor saved to: {files} [/green]")

print()

if load_dataset:
    print("Loading dataset from:", dataset_path)
    dataset = Dataset.load(dataset_path, validate_shape=True)
    print("[green]Dataset loaded successfully.[/green]")
else:
    print("preprocessing dataset...")
    dataset = pp.preprocess_dataset(
        logs_per_class=ds_logs_per_class, 
        shuffle=ds_shuffle, 
        max_events=ds_max_events
        )
    print("[green]Dataset preprocessed successfully.[/green]")
    print("Saving dataset to:", dataset_path)
    dataset.save(dataset_path, True)

print()

print("Splitting dataset into train and test sets...")
train, test = dataset.stratified_split((4, 1))
X_train, y_train = train
X_test, y_test = test

print("Train X shape:", X_train.shape, "Test X shape:", X_test.shape)
print("Train y shape:", y_train.shape, "Test y shape:", y_test.shape)

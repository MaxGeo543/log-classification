from lstm import LSTMClassifier
from preprocessor import Preprocessor
from keras.models import Sequential
from keras import Input, Model, layers

from util import get_sorted_log_numbers_by_size

from dataset import Dataset
from pseudolabeling import DynamicDataset, PseudoLabelingCallback
from keras.callbacks import EarlyStopping


log_file_pattern = "./data/CCI/CCLog-backup.{num}.log"
logs_to_load = get_sorted_log_numbers_by_size("./data/CCI")[:20]

# get preprocessor
preprocessor_file = ""
if preprocessor_file:
    pp = Preprocessor.load(preprocessor_file)
else:
    from encoders.datetime_encoder import DatetimeEncoder
    from encoders.datetime_features import DatetimeFeature, DatetimeFeatureBase
    from encoders.loglevel_encoder import *
    from encoders.message_encoder import *
    from encoders.function_encoder import *
    from encoders.classes_encoder import *
    from classes import classes, annotate as cl_annotate

    pp_window_size = 20
    pp_classes = classes
    annotation_func = cl_annotate
    pp_classes_encoder = ClassesLabelBinarizer()
    pp_log_level_encoder = LogLevelOrdinalEncoder()
    pp_datetime_encoder = DatetimeEncoder([DatetimeFeature.second.since_midnight, DatetimeFeature.day.since_epoch])
    pp_function_encoder = FunctionOneHotEncoder(min_frequency=3, max_categories=1000)
    pp_message_encoder = MessageTextVectorizationEncoder(max_tokens=1000, output_mode="count")

    pp = Preprocessor(
        message_encoder=pp_message_encoder,
        function_encoder=pp_function_encoder,
        datetime_encoder=pp_datetime_encoder,
        log_level_encoder=pp_log_level_encoder,
        classes_encoder=pp_classes_encoder,
        classes=pp_classes,
        window_size=pp_window_size,
        name="train_lstm_pp",
        volatile=True
    )

# create dataset
data_file = ""
if data_file:
    data = Dataset.load(data_file)
    if not pp.encoders_initialized: 
        pp.initialize_encoders()
else:
    pp.load_logfiles([log_file_pattern.format(num=num) for num in logs_to_load])
    pp.initialize_encoders()
    print(len(pp.events))
    dataset = pp.preprocess_dataset(
        logs_per_class=100,
        shuffle=True,
        force_logs_per_class=False,
        max_events=None
    )
    
    print(len(dataset.data_list))

# define model
model = LSTMClassifier(
    preprocessor=pp,
    dataset=dataset,
    
    data_split_ratios=(4, 1, 1),  # Train, validation, test split ratios
)

# train model
model.train(epochs=1000, batch_size=32)
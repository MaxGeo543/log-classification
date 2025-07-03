from preprocessor2 import Preprocessor
from message_encoder import *
from util import get_sorted_log_numbers_by_size

s = get_sorted_log_numbers_by_size(r"D:\mgeo\projects\log-classification\data\CCI")

# preprocessing parameters
log_files = s[0:20:4] # [i for i in range(745, 773)]             # list of ints representing the numbers of log files to use
logs_per_class = 100                                 # How many datapoints per class should be collected if available
window_size = 20                                     # how many log messages to be considered in a single data point from sliding window
encoding_output_size = 16                            # size to be passed to the message_encoder, note that this is not neccessairily the shape of the output
message_encoder = BERTEncoder(encoding_output_size)  # the message_encoder to be used. Can be TextVectorizationEncoder (uses keras.layers.TextVectorizer), BERTEncoder (only uses the BERT tokenizer) or BERTEmbeddingEncoder (also uses the BERT model)
extended_datetime_features = False                   # bool, whether the preprocessing should use a multitude of normalized features extracted from the date 

# preprocessing
pp = Preprocessor(log_files, message_encoder, logs_per_class, window_size, extended_datetime_features, volatile=True)
pp.preprocess(ignore_max_logs_per_class=True)
n = f"{pp.logs_per_class}lpc_{pp.window_size}ws_bertencx{pp.message_encoder.get_result_shape() if pp.message_encoder is not None else ''}"
pp.save(f"./data/preprocessor_5_files_all_data_{pp.logs_per_class}lpc_{pp.window_size}ws_bertencx{pp.message_encoder.get_result_shape() if pp.message_encoder is not None else ''}")

from preprocessor2 import Preprocessor
from function_encoder import *
from message_encoder import *
from util import get_sorted_log_numbers_by_size

s = get_sorted_log_numbers_by_size(r"D:\mgeo\projects\log-classification\data\CCI")

# preprocessing parameters
log_files = s[:20] # [i for i in range(745, 773)]             # list of ints representing the numbers of log files to use
logs_per_class = 100                                 # How many datapoints per class should be collected if available
window_size = 20                                     # how many log messages to be considered in a single data point from sliding window
encoding_output_size = 16                            # size to be passed to the message_encoder, note that this is not neccessairily the shape of the output
message_encoder = BERTEncoder(encoding_output_size)  # the message_encoder to be used. Can be TextVectorizationEncoder (uses keras.layers.TextVectorizer), BERTEncoder (only uses the BERT tokenizer) or BERTEmbeddingEncoder (also uses the BERT model)
extended_datetime_features = False                   # bool, whether the preprocessing should use a multitude of normalized features extracted from the date 

# preprocessing
with open("./data/unique_methods.txt") as f:
    lines = [line.rstrip('\n') for line in f.readlines()]
    function_encoder = FunctionOrdinalEncoder()
    function_encoder.initialize(lines)
pp = Preprocessor(log_files, message_encoder, logs_per_class, window_size, extended_datetime_features, function_encoder=function_encoder, volatile=True)
pp.preprocess()


pp.save()

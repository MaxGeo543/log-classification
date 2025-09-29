from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from log_classification.util import hash_list_to_string

# FunctionEncoder Base
class LogLevelEncoder(ABC):
    def __init__(self):
        self.initialized = False
    
    @abstractmethod
    def initialize(self, all_levels):
        """
        Initialize the LogLevelEncoder with all log levels
        """
        raise NotImplementedError()

    @abstractmethod
    def encode(self, log_level: str):
        """
        Encode a single log levels
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_dimension(self):
        """
        Get the output dimension of log level encodings
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_key(self):
        """
        Get a key unique to the Encoder
        """
        raise NotImplementedError()

class LogLevelOneHotEncoder(LogLevelEncoder):
    """
    Encode log levels into a One-Hot Encoded Vector 
    Uses sklearn.preprocessing.OneHotEncoder
    """
    def __init__(self, min_frequency: int = 2, max_categories: int | None = None):
        """
        :params min_frequency: How often a function must appear in the initialization data to be considered a function
        :params max_categories: The maximum amount of categories
        """
        super().__init__()
        self.one_hot_encoder = OneHotEncoder(sparse_output=True, handle_unknown="infrequent_if_exist")
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.one_hot_encoder = OneHotEncoder(
            min_frequency=min_frequency, 
            max_categories=max_categories,
            handle_unknown="infrequent_if_exist",
            sparse_output=True)
    
    def initialize(self, all_levels: list[str]):
        self.one_hot_encoder.fit([[ll] for ll in all_levels])
        self.initialized = True

    def encode(self, log_level: str):
        return self.one_hot_encoder.transform([[log_level]]).toarray().flatten()
    
    def get_dimension(self):
        return self.one_hot_encoder.transform([["dummy"]]).shape[1]
    
    def get_key(self):
        key = hash_list_to_string([
            "LogLevelOneHotEncoder",
            str(self.min_frequency),
            str(self.max_categories),
            *[str(word) for cat in self.one_hot_encoder.categories_ for word in cat]
        ], 16)
        return key

class LogLevelOrdinalEncoder(LogLevelEncoder):
    """
    Encode categorical features as an integer array. Encode classes with value between 0 and n_classes-1. Functionally the same as LabelEncoder.
    Unrecognized values will be encoded as -1.
    This uses the sklearn.preprocessing.OrdinalEncoder
    """
    def __init__(self):
        super().__init__()
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    def initialize(self, all_levels: list[str]):
        self.ordinal_encoder.fit([[l] for l in all_levels])
        self.initialized = True
    
    def encode(self, log_level):
        return self.ordinal_encoder.transform([[log_level]])[0]
    
    def get_dimension(self):
        return 1
    
    def get_key(self):
        key = hash_list_to_string([
            "LogLevelOrdinalEncoderEncoder",
            *[str(word) for word in self.ordinal_encoder.categories_]
        ], 16)
        return key


if __name__ == "__main__":
    x = LogLevelOrdinalEncoder()
    x.initialize(["a", "b", "c"])
    print(x.get_dimension())
    print(x.encode("c"))
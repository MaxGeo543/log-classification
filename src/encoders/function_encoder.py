from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from util import hash_list_to_string

# FunctionEncoder Base
class FunctionEncoder(ABC):
    def __init__(self):
        self.initialized = False
    
    @abstractmethod
    def initialize(self, all_functions):
        """
        Initialize the FunctionEncoder with all functions
        """
        raise NotImplementedError()

    @abstractmethod
    def encode(self, function: str):
        """
        Encode a single function
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_dimension(self):
        """
        Get the output dimension of function encodings
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_key(self):
        """
        Get a key unique to the Encoder
        """
        raise NotImplementedError()

class FunctionLabelEncoder(FunctionEncoder):
    """
    Encode functions with value between 0 and n_functions-1.
    This uses the sklearn.preprocessing.LabelEncoder, explicitely stating to be used with Output, not input, therefore it is recommended to use 
    FunctionOrdinalEncoder instead
    """
    def __init__(self):
        super().__init__()
        self.label_encoder = LabelEncoder()
    
    def initialize(self, all_functions):
        self.label_encoder.fit(all_functions)
        self.initialized = True
    
    def encode(self, function):
        return self.label_encoder.transform([function])[0]
    
    def get_dimension(self):
        return 1
    
    def get_key(self):
        key = hash_list_to_string([
            "FunctionLabelEncoder",
            *[str(word) for word in self.label_encoder.classes_]
        ], 16)
        return key

class FunctionOrdinalEncoder(FunctionEncoder):
    """
    Encode categorical features as an integer array. Encode classes with value between 0 and n_classes-1. Functionally the same as LabelEncoder.
    Unrecognized values will be encoded as -1.
    This uses the sklearn.preprocessing.OrdinalEncoder
    """
    def __init__(self):
        super().__init__()
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    def initialize(self, all_functions):
        # OrdinalEncoder expects a list of all function strings
        self.ordinal_encoder.fit([[f] for f in all_functions])
        self.initialized = True
    
    def encode(self, function):
        # Transform a single value, input must be 2D
        return self.ordinal_encoder.transform([[function]])[0]

    def get_dimension(self):
        return 1
    
    def get_key(self):
        key = hash_list_to_string([
            "FunctionOrdinalEncoder",
            *[str(word) for cat in self.ordinal_encoder.categories_ for word in cat]
        ], 16)
        return key

class FunctionOneHotEncoder(FunctionEncoder):
    """
    Encode functions into a One-Hot Encoded Vector 
    Uses sklearn.preprocessing.OneHotEncoder
    """
    def __init__(self, min_frequency: int = 2, max_categories: int | None = None):
        """
        :params min_frequency: How often a function must appear in the initialization data to be considered a function
        :params max_categories: The maximum amount of categories
        """
        
        super().__init__()
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.one_hot_encoder = OneHotEncoder(
            min_frequency=min_frequency, 
            max_categories=max_categories,
            handle_unknown="infrequent_if_exist",
            sparse_output=True)
    
    def initialize(self, all_functions: list[str]):
        self.one_hot_encoder.fit([[f] for f in all_functions])
        self.initialized = True

    def encode(self, function: str):
        return self.one_hot_encoder.transform([[function]]).toarray().flatten()
    
    def get_dimension(self):
        return self.one_hot_encoder.transform([["dummy"]]).shape[1]
    
    def get_key(self):
        key = hash_list_to_string([
            "FunctionOneHotEncoder",
            str(self.min_frequency),
            str(self.max_categories),
            *[str(word) for cat in self.one_hot_encoder.categories_ for word in cat]
        ], 16)
        return key


if __name__ == "__main__":
    x = FunctionOrdinalEncoder()
    x.initialize(["a", "b", "c"])
    print(x.get_dimension())
    print(x.encode("d"))
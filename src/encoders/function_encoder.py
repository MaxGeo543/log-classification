from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from hash_list import hash_list_to_string

# FunctionEncoder Base
class FunctionEncoder(ABC):
    def __init__(self):
        self.initialized = False
    
    @abstractmethod
    def initialize(self, all_functions):
        raise NotImplementedError()

    @abstractmethod
    def encode(self, function: str):
        raise NotImplementedError()
    
    @abstractmethod
    def get_dimension(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_key(self):
        raise NotImplementedError()

class FunctionLabelEncoder(FunctionEncoder):
    def __init__(self):
        super().__init__()
        self.label_encoder = LabelEncoder()
    
    def initialize(self, all_functions):
        self.label_encoder.fit(all_functions)
    
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
    def __init__(self):
        super().__init__()
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    def initialize(self, all_functions):
        # OrdinalEncoder expects a list of all function strings
        self.ordinal_encoder.fit([[f] for f in all_functions])
        self.initialized = True
    
    def encode(self, function):
        # Transform a single value, input must be 2D
        return int(self.ordinal_encoder.transform([[function]])[0][0])

    def get_dimension(self):
        return 1
    
    def get_key(self):
        key = hash_list_to_string([
            "FunctionOrdinalEncoder",
            *[str(word) for cat in self.ordinal_encoder.categories_ for word in cat]
        ], 16)
        return key

class FunctionOneHotEncoder(FunctionEncoder):
    def __init__(self, min_frequency: int = 2, max_categories: int | None = None):
        super().__init__()
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.one_hot_encoder = OneHotEncoder(
            min_frequency=min_frequency, 
            max_categories=max_categories,
            sparse_output=True)
    
    def initialize(self, all_functions: list[str]):
        self.one_hot_encoder.fit([[f] for f in all_functions])
        self.initialized = True

    def encode(self, function: str):
        return self.one_hot_encoder.transform([[function]]).flatten()
    
    def get_dimension(self):
        return len(self.one_hot_encoder.categories_[0])
    
    def get_key(self):
        key = hash_list_to_string([
            "FunctionOneHotEncoder",
            str(self.min_frequency),
            str(self.max_categories),
            *[str(word) for cat in self.one_hot_encoder.categories_ for word in cat]
        ], 16)
        return key
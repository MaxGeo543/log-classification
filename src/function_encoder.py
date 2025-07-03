from abc import ABC, abstractmethod
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

class FunctionEncoder(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def initialize(self, all_functions):
        pass

    @abstractmethod
    def encode(self, function: str):
        raise NotImplementedError()
    

class FunctionLabelEncoder(FunctionEncoder):
    def __init__(self):
        self.label_encoder = LabelEncoder()
    
    def initialize(self, all_functions):
        self.label_encoder.fit(all_functions)
    
    def encode(self, function):
        return self.label_encoder.transform([function])[0]

class FunctionOrdinalEncoder(FunctionEncoder):
    def __init__(self):
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    def initialize(self, all_functions):
        # OrdinalEncoder expects a 2D list of shape (n_samples, n_features)
        self.ordinal_encoder.fit([[f] for f in all_functions])
    
    def encode(self, function):
        # Transform a single value, input must be 2D
        return int(self.ordinal_encoder.transform([[function]])[0][0])
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from hash_list import hash_list_to_string

# FunctionEncoder Base
class LogLevelEncoder(ABC):
    def __init__(self):
        self.initialized = False
    
    @abstractmethod
    def initialize(self, all_levels):
        raise NotImplementedError()

    @abstractmethod
    def encode(self, log_level: str):
        raise NotImplementedError()
    
    @abstractmethod
    def get_dimension(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_key(self):
        raise NotImplementedError()

class LogLevelOneHotEncoder(LogLevelEncoder):
    def __init__(self):
        super().__init__()
        self.one_hot_encoder = OneHotEncoder(sparse_output=True, handle_unknown="infrequent_if_exist")
    
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
            *[str(word) for cat in self.one_hot_encoder.categories_ for word in cat]
        ], 16)
        return key

class LogLevelOrdinalEncoder(LogLevelEncoder):
    def __init__(self):
        super().__init__()
        self.ordinal_encoder = OrdinalEncoder()
    
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
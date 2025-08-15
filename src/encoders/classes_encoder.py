from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from abc import ABC, abstractmethod
from hash_list import hash_list_to_string
import numpy as np

class ClassesEncoder(ABC):
    def __init__(self):
        self.initialized = False
    
    @abstractmethod
    def initialize(self, all_labels):
        raise NotImplementedError()

    @abstractmethod
    def encode(self, label: str):
        raise NotImplementedError()
    
    @abstractmethod
    def get_dimension(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_key(self):
        raise NotImplementedError()

class ClassesLabelEncoder(ClassesEncoder):
    def __init__(self):
        super().__init__()
        self.label_encoder = LabelEncoder()
    
    def initialize(self, all_labels):
        self.label_encoder.fit(all_labels)
        self.initialized = True
    
    def encode(self, label):
        return self.label_encoder.transform([label])
    
    def get_dimension(self):
        return 1
    
    def get_key(self):
        key = hash_list_to_string([
            "ClassesLabelEncoder",
            *[str(word) for word in self.label_encoder.classes_]
        ], 16)
        return key

class ClassesLabelBinarizer(ClassesEncoder):
    def __init__(self):
        super().__init__()
        self.one_hot_encoder = LabelBinarizer(sparse_output=True)
    
    def initialize(self, all_labels: list[str]):
        self.one_hot_encoder.fit([l for l in all_labels])
        self.initialized = True

    def encode(self, label: str):
        if len(self.one_hot_encoder.classes_) == 1:
            correct = self.one_hot_encoder.classes_[0]
            return np.array([1 if label == correct else 0])
        elif len(self.one_hot_encoder.classes_) == 2:
            return np.array([1 if label == correct else 0 for correct in self.one_hot_encoder.classes_])
        
        return self.one_hot_encoder.transform([label]).toarray().flatten()
    
    def get_dimension(self):
        return len(self.one_hot_encoder.classes_)
    
    def get_key(self):
        key = hash_list_to_string([
            "ClassesLabelBinarizer",
            *[str(word) for word in self.one_hot_encoder.classes_]
        ], 16)
        return key

if __name__ == "__main__":
    x = ClassesLabelBinarizer()
    x.initialize(["A", "Z"])
    print(x.get_dimension())
    print(x.encode("A"))
    print(x.initialized)
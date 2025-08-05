from abc import ABC, abstractmethod
import torch
from keras.layers import TextVectorization
from transformers import BertTokenizer
from transformers import BertModel
from typing import Any
from hash_list import hash_list_to_string

# Standardization strategy Base
class Standardization(ABC):
    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        return super().__str__()

# Split strategy Base
class Split(ABC):
    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplementedError()
    
    @abstractmethod
    def __str__(self):
        return super().__str__()


# MessageEncoder Base
class MessageEncoder(ABC):
    def __init__(self):
        self.initialized = False
    
    @abstractmethod
    def initialize(self, all_messages):
        raise NotImplementedError()

    @abstractmethod
    def encode(self, message: str):
        raise NotImplementedError()
    
    @abstractmethod
    def get_dimension(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_key(self):
        raise NotImplementedError()

class TextVectorizationEncoder(MessageEncoder):
    def __init__(self, 
                 max_tokens: int = 10_000,
                 standardize: str | Standardization = "lower_and_strip_punctuation",
                 split: str | Split = "whitespace",
                 output_mode: str = "int",
                 output_sequence_length: int = 16):
        super().__init__()
        # set members of class
        self.max_tokens = max_tokens
        self.standardize = standardize
        self.split = split
        self.output_mode = output_mode
        self.output_sequence_length = output_sequence_length
        # define the text vectorizer
        self._text_vectorizer = TextVectorization(
            max_tokens=max_tokens,
            standardize=standardize,
            split=split,
            output_mode=output_mode,
            output_sequence_length=output_sequence_length
        )
    
    def initialize(self, all_messages):
        self._text_vectorizer.adapt(all_messages)
        self.initialized = True

    def encode(self, message):
        return self._text_vectorizer(message)
    
    def get_dimension(self) -> int:
        if self._text_vectorizer._output_mode == "int":
            return self._text_vectorizer._output_sequence_length
        else:
            return len(self._text_vectorizer.get_vocabulary())

    def get_key(self):
        key = hash_list_to_string([
            "TextVectorization",
            str(self.max_tokens),
            str(self.standardize),
            str(self.split),
            self.output_mode,
            str(self.output_sequence_length),
            *self._text_vectorizer.get_vocabulary()
        ], 16)
        return key

class BERTEncoder(MessageEncoder):
    def __init__(self, 
                 max_length=16):
        super().__init__()
        self.max_length = max_length
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def initialize(self, all_messages):
        pass

    def encode(self, message):
        # encode tokens
        inputs = self._tokenizer(
            message,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        x = inputs['input_ids'][0]
        return x[:self.max_length]
    
    def get_dimension(self):
        return self.max_length
    
    def get_key(self):
        key = hash_list_to_string([
            "BERTEncoder",
            str(self.max_length)
        ], 16)
        return key

class BERTEmbeddingEncoder(MessageEncoder):
    def __init__(self, max_length = 16, output_mode: str = "all"):
        super().__init__()
        # set members
        self.max_length = max_length
        self.output_mode = output_mode
        if self.output_mode not in ["all", "cls"]: raise ValueError("output_mode must be 'all' or 'cls'")
        # initialize tokenizer
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # initialize model
        self._model = BertModel.from_pretrained('bert-base-uncased')
        self._model.eval()  # Set model to evaluation mode
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def initialize(self, all_messages):
        self.initialized = True

    def encode(self, message):
        # encode tokens
        inputs = self._tokenizer(
            message,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        # embed using the model
        with torch.no_grad():
            outputs = self._model(**inputs)

        if self.output_mode == "cls":
            # Extract CLS token embedding: shape (1, hidden_size)
            embeddings = outputs.last_hidden_state[:, 0, :]
            # Squeeze to shape: (hidden_size)
            return embeddings.squeeze(0)
        elif self.output_mode == "all":
            # Shape before flattening: (max_length, hidden_size)
            embeddings = outputs.last_hidden_state[0]
            # Flatten to shape: (hidden_size * max_len)
            return embeddings.flatten()
        else: raise ValueError("output_mode must be 'all' or 'cls'")

    def get_dimension(self):
        if self.output_mode == "cls":
             # Return shape: hidden_dim (768)
            return self._model.config.hidden_size
        elif self.output_mode == "all":
             # Return shape: sequence_length * hidden_dim, e.g., 16 * 768
            return self.max_length * self._model.config.hidden_size
        else: raise ValueError("output_mode must be 'all' or 'cls'")

    def get_key(self):
        key = hash_list_to_string([
            "BERTEmbeddingEncoder",
            str(self.max_length),
            self.output_mode
        ], 16)
        return key
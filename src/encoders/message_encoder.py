from abc import ABC, abstractmethod
import torch
from keras.layers import TextVectorization
from transformers import BertTokenizer
from transformers import BertModel
from typing import Any
from util import hash_list_to_string

# Standardization strategy Base
class Standardization(ABC):
    """
    A standardization method for messages, must be callable and convertable to string
    """
    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        return super().__str__()

# Split strategy Base
class Split(ABC):
    """
    A split methodstrategy for messages, must be callable and convertable to string
    """
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
        """
        Initialize the MessageEncoder with all messages
        """
        raise NotImplementedError()

    @abstractmethod
    def encode(self, message: str):
        """
        Encode a single message
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_dimension(self):
        """
        Get the output dimension of message encodings
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_key(self):
        """
        Get a key unique to the Encoder
        """
        raise NotImplementedError()

class MessageTextVectorizationEncoder(MessageEncoder):
    """
    Encode messages using TextVectorization, maps text features to integer sequences.

    uses keras.Layers.TextVectorization
    """
    def __init__(self, 
                 max_tokens: int = 10_000,
                 standardize: str | Standardization = "lower_and_strip_punctuation",
                 split: str | Split = "whitespace",
                 output_mode: str = "int",
                 output_sequence_length: int = 16):
        """
        :params max_tokens: Maximum size of the vocabulary
        :params standardize: can be "lower_and_strip_punctuation", "lower", "strip_punctuation" or any callable, inputs will be passed to the callable and should be standardized and returned
        :params split: can be "whitespace", "character" or any callable, the standardized input will be passed to the callable which should split and return it
        :params output_mode: can be "int", "multi_hot", "count" or "tf_idf", for more information look at the documentation of keras.Layers.TextVectorization
        :params output_sequence_length: is only used if output mode is int, sets the dimension of the output encodings
        """
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
            output_sequence_length=output_sequence_length if output_mode == "int" else None
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

class MessageBERTEncoder(MessageEncoder):
    """
    Uses a pretrained vocabulary to tokenize the input messages as it is done for BERT
    
    uses transformers.BertTokenizer
    """
    def __init__(self, 
                 max_length=16):
        super().__init__()
        self.max_length = max_length
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.initialized = True
    
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

class MessageBERTEmbeddingEncoder(MessageEncoder):
    """
    uses the pretrained BERT model to encode text features

    uses transformers.BertModel
    """
    def __init__(self, max_length = 16, output_mode: str = "all"):
        """
        :params max_length: only used  if output_mode is "all", 
        :params output_mode: "all" or "cls", if it is "cls" only the cls encoding will be returned which has dimension hidden_dim (768), otherwise all encodings will be returned with dimension (max_length * hidden_dim)
        """
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
from abc import ABC, abstractmethod
from keras.layers import TextVectorization
from transformers import BertTokenizer

class MessageEncoder(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def initialize(self, all_messages):
        pass

    @abstractmethod
    def encode(self, message: str):
        raise NotImplementedError()

class TextVectorizationEncoder(MessageEncoder):
    def __init__(self, 
                 max_tokens=10000,
                 pad_to_max_tokens=True,
                 output_sequence_length=1):
        super().__init__()
        self._text_vectorizer = TextVectorization(
            max_tokens=max_tokens,
            output_mode='int',
            pad_to_max_tokens=pad_to_max_tokens,
            output_sequence_length=output_sequence_length
        )
    
    def initialize(self, all_messages):
        self._text_vectorizer.adapt(all_messages)

    def encode(self, message):
        return self._text_vectorizer(message)


class BERTEncoder(MessageEncoder):
    def __init__(self, 
                 max_length=16):
        super().__init__()
        self.max_length = max_length
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def initialize(self, all_messages):
        pass

    def encode(self, message):
        inputs = self._tokenizer(
            message,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        return inputs['input_ids'][0]
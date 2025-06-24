from abc import ABC, abstractmethod
import torch
from keras.layers import TextVectorization
from transformers import BertTokenizer
from transformers import BertModel

class MessageEncoder(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def initialize(self, all_messages):
        pass

    @abstractmethod
    def encode(self, message: str):
        raise NotImplementedError()
    
    @abstractmethod
    def get_result_shape(self):
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
    
    def get_result_shape(self):
        return self._text_vectorizer._output_sequence_length



class BERTEncoder(MessageEncoder):
    def __init__(self, 
                 max_length=16):
        super().__init__()
        self.max_length = max_length
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', )
    
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
        x = inputs['input_ids'][0]
        # print(x.shape)
        return x[:self.max_length]
    
    def get_result_shape(self):
        return self.max_length


class BERTEmbeddingEncoder(MessageEncoder):
    def __init__(self, max_length=16):
        super().__init__()
        self.max_length = max_length
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained('bert-base-uncased')
        self._model.eval()  # Set model to evaluation mode
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def initialize(self, all_messages):
        pass  # No initialization needed in this case

    def encode(self, message):
        inputs = self._tokenizer(
            message,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Shape before flattening: (max_length, hidden_size)
        embeddings = outputs.last_hidden_state[0]

        # Flatten to shape: (max_length * hidden_size,)
        return embeddings.flatten()

    def get_result_shape(self):
        # Return shape: sequence_length * hidden_dim, e.g., 16 * 768
        return self.max_length * self._model.config.hidden_size
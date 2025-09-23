from keras.layers import Dense, GlobalAveragePooling1D

from keras import layers
from keras.saving import register_keras_serializable
from positional_encoding import PositionalEncoding


@register_keras_serializable(package="custom")
class TransformerBlock(layers.Layer):
    """
    A single Transformer encoder block: (Post-LN) Self-Attention + FFN.
    """
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)

        self.attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.model_dim // self.num_heads,
            dropout=self.dropout,
        )
        self.dropout1 = layers.Dropout(self.dropout)
        self.add1 = layers.Add()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn_dense1 = layers.Dense(self.ff_dim, activation="relu")
        self.ffn_dense2 = layers.Dense(self.model_dim)
        self.dropout2 = layers.Dropout(self.dropout)
        self.add2 = layers.Add()
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.supports_masking = True

    def call(self, x, training=None, mask=None):
        # Self-attention (use the incoming sequence mask if provided)
        attn_out = self.attn(x, x, attention_mask=mask, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.add1([x, attn_out])
        x = self.norm1(x)

        # Feed-forward
        ff = self.ffn_dense1(x)
        ff = self.ffn_dense2(ff)
        ff = self.dropout2(ff, training=training)
        x = self.add2([x, ff])
        x = self.norm2(x)
        return x

    def compute_mask(self, inputs, mask=None):
        # Keep the same mask shape for downstream layers.
        return mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
            }
        )
        return config

@register_keras_serializable(package="custom")
class TransformerClassifierLayer(layers.Layer):
    """
    A reusable Transformer encoder stack + classifier, implemented as a Keras Layer.
    Plug into Functional or Sequential models.

    Pipeline:
      Dense(model_dim) -> (optional) PositionalEmbedding ->
      [TransformerBlock] * transformer_layers ->
      GlobalAveragePooling1D -> Dense(hidden_units, relu) -> Dense(num_classes, softmax)
    """
    def __init__(
        self,
        num_classes: int,

        transformer_layers: int = 2,
        model_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        hidden_units: int = 64,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.transformer_layers = int(transformer_layers)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.hidden_units = int(hidden_units)

        self.supports_masking = True

        # Input projection to model dimension
        self._proj = Dense(self.model_dim)

        # Optional positional embedding
        self._pos = PositionalEncoding()


        # Transformer stack
        self._blocks = [
            TransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
            )
            for _ in range(self.transformer_layers)
        ]

        # Pool + classifier head
        self._gap = GlobalAveragePooling1D()
        self._hidden = Dense(self.hidden_units, activation="relu")
        self._classifier = Dense(self.num_classes, activation="softmax")

    def call(self, inputs, training = None, mask=None):
        x = self._proj(inputs)
        if self._pos is not None:
            x = self._pos(x)  # adds positional embeddings

        for block in self._blocks:
            x = block(x, training=training, mask=mask)

        x = self._gap(x, mask=mask)
        x = self._hidden(x)
        return self._classifier(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "transformer_layers": self.transformer_layers,
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
                "max_length": self.max_length,
                "hidden_units": self.hidden_units,
            }
        )
        return config
"""PsyCNN (Table 4) and PsyTransformer (Tables 5-6) architectures."""


from .compat import tf, regularizers, Activation, Add, BatchNormalization, Conv1D, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, LayerNormalization, MaxPooling1D, Model
from .config import L2_REG


def residual_block(x, filters: int, dropout: float, l2: float):
    shortcut = x
    x = Conv1D(filters, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv1D(filters, 1, padding="same", kernel_regularizer=regularizers.l2(l2))(shortcut)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    x = MaxPooling1D(2)(x)
    return Dropout(dropout)(x)


def build_ti_cnn(t: int, d: int, n_classes: int, l2: float = L2_REG) -> Model:
    inp = Input(shape=(t, d), name="features")
    x = Conv1D(48, 5, padding="same", kernel_regularizer=regularizers.l2(l2))(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = residual_block(x, 80, 0.25, l2)
    x = residual_block(x, 112, 0.30, l2)
    x = residual_block(x, 160, 0.35, l2)
    frame = Conv1D(64, 1, padding="same", activation="relu", name="frame_proj")(x)
    va_frame = Conv1D(2, 1, padding="same", name="va_frame")(frame)
    pooled = GlobalAveragePooling1D(name="cnn_global_pool")(x)
    hidden = Dense(192, activation="relu", kernel_regularizer=regularizers.l2(l2), name="classifier_hidden")(pooled)
    hidden = Dropout(0.5, name="classifier_dropout")(hidden)
    cls = Dense(n_classes, activation="softmax", name="cls", kernel_regularizer=regularizers.l2(l2))(hidden)
    va_utt = GlobalAveragePooling1D(name="va_utt")(va_frame)
    return Model(inp, [cls, va_frame, va_utt], name="TI_CNN")


def transformer_block(x, num_heads: int, key_dim: int, ff_dim: int, dropout: float = 0.1):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn)
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(int(x.shape[-1]))(ff)
    return LayerNormalization(epsilon=1e-6)(x + ff)


def build_ti_transformer(t: int, d: int, n_classes: int, d_model: int = 128,
                         num_layers: int = 3, num_heads: int = 4,
                         ff_dim: int = 256, l2: float = L2_REG) -> Model:
    inp = Input(shape=(t, d), name="features")
    x = Dense(d_model, name="input_projection")(inp)

    # Fixed positional embedding (broadcastable to any batch size)
    pos_ids = tf.range(start=0, limit=t, delta=1)
    positions = Embedding(input_dim=t, output_dim=d_model, name="position_embedding")(pos_ids)
    positions = tf.expand_dims(positions, axis=0)   # (1, t, d_model)
    x = Add(name="add_position_embedding")([x, positions])

    for i in range(num_layers):
        x = transformer_block(x, num_heads, d_model // num_heads, ff_dim, dropout=0.1)
    va_frame = Dense(2, name="va_frame")(x)
    pooled = GlobalAveragePooling1D(name="transformer_global_pool")(x)
    hidden = Dense(192, activation="relu", kernel_regularizer=regularizers.l2(l2), name="classifier_hidden")(pooled)
    hidden = Dropout(0.4, name="classifier_dropout")(hidden)
    cls = Dense(n_classes, activation="softmax", name="cls", kernel_regularizer=regularizers.l2(l2))(hidden)
    va_utt = GlobalAveragePooling1D(name="va_utt")(va_frame)
    return Model(inp, [cls, va_frame, va_utt], name="TI_Transformer")

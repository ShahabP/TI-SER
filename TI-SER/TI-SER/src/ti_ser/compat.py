"""Optional third-party imports with graceful fallbacks.

``ensure_dependencies()`` (see ``utils``) raises a clear error before any
experiment starts if a required package is missing. Keeping the imports here
lets ``ti_ser --help`` and the lightweight unit tests run without TensorFlow.
"""
import numpy as np

# Matplotlib is deliberately non-interactive for reliable batch/Colab exports.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # checked with a helpful message in ensure_dependencies()
    plt = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
except ImportError:
    accuracy_score = balanced_accuracy_score = classification_report = None
    confusion_matrix = f1_score = precision_recall_fscore_support = None
    GroupShuffleSplit = train_test_split = TSNE = StandardScaler = None

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import (
        Activation,
        Add,
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        Embedding,
        GlobalAveragePooling1D,
        Input,
        LayerNormalization,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
except ImportError:
    tf = K = regularizers = None
    Activation = Add = BatchNormalization = Conv1D = Dense = Dropout = None
    Embedding = GlobalAveragePooling1D = Input = LayerNormalization = None
    MaxPooling1D = Model = Adam = to_categorical = None

__all__ = ['np', 'plt', 'librosa', 'accuracy_score', 'balanced_accuracy_score', 'classification_report', 'confusion_matrix', 'f1_score', 'precision_recall_fscore_support', 'GroupShuffleSplit', 'train_test_split', 'TSNE', 'StandardScaler', 'tf', 'K', 'regularizers', 'Activation', 'Add', 'BatchNormalization', 'Conv1D', 'Dense', 'Dropout', 'Embedding', 'GlobalAveragePooling1D', 'Input', 'LayerNormalization', 'MaxPooling1D', 'Model', 'Adam', 'to_categorical']

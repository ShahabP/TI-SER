"""Manuscript configuration: feature settings, loss weights, prototypes, label maps, default seeds/epochs."""


from .compat import np


SR = 22_050


N_MFCC = 40


MAX_FRAMES = 174


D_FEATURES = N_MFCC * 3


N_CLASSES = 7


EMOTIONS = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]


CREMAD_EMOTIONS = ["angry", "happy", "sad", "neutral", "fear", "disgust"]


BATCH_SIZE = 32


INITIAL_LR = 1e-3


L2_REG = 1e-4


AUGMENT_TIMES = 2


FOCAL_ALPHA = 0.2


FOCAL_GAMMA = 2.0


LAMBDA_CE = 1.0


LAMBDA_FOCAL = 1.0


LAMBDA_DIM = 0.5


LAMBDA_INT = 0.3


LAMBDA_PHASE = 0.2


PHASE_WINDOW = 0.10


PHASE_SLOPE_WEIGHT = 0.5


PHASE_CURVATURE_WEIGHT = 0.1


# Match Table 3 in the supplied manuscript: neutral VA=(0.0, 0.5).
# These values must be rerun—not retroactively assigned—to any old results.
VA_PROTOTYPES = np.asarray(
    [
        [-0.50, 0.70],  # angry
        [0.80, 0.70],   # happy
        [-0.70, 0.30],  # sad
        [0.00, 0.50],   # neutral (matches manuscript Table 3)
        [-0.60, 0.80],  # fear
        [-0.60, 0.60],  # disgust
        [0.50, 0.90],   # surprise
    ],
    dtype=np.float32,
)


TESS_MAP = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3,
    "fear": 4,
    "disgust": 5,
    "ps": 6,
    "pleasant_surprise": 6,
    "surprise": 6,
}


RAV_MAP = {1: 3, 2: 3, 3: 1, 4: 2, 5: 0, 6: 4, 7: 5, 8: 6}


CREMA_MAP = {"ang": 0, "hap": 1, "sad": 2, "neu": 3, "fea": 4, "dis": 5}


DEFAULT_SEEDS = [42, 43, 44, 45, 46]


DEFAULT_ABLATION_SEEDS = [42, 43, 44]


DEFAULT_SENSITIVITY_SEEDS = [42, 43, 44]


DEFAULT_EPOCHS = 20


DEFAULT_ABLATION_EPOCHS = 15


DEFAULT_SENSITIVITY_EPOCHS = 15

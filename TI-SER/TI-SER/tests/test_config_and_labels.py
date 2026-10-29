"""Checks that configuration and label maps match the manuscript."""
import numpy as np
import pytest

from ti_ser import config
from ti_ser.corpora import label_from_path


def test_manuscript_constants():
    assert config.SR == 22_050
    assert config.N_MFCC == 40 and config.D_FEATURES == 120
    assert config.MAX_FRAMES == 174
    assert config.N_CLASSES == 7
    assert config.BATCH_SIZE == 32
    assert config.INITIAL_LR == 1e-3 and config.L2_REG == 1e-4
    assert config.AUGMENT_TIMES == 2
    assert (config.FOCAL_ALPHA, config.FOCAL_GAMMA) == (0.2, 2.0)
    assert (config.LAMBDA_CE, config.LAMBDA_FOCAL) == (1.0, 1.0)
    assert (config.LAMBDA_DIM, config.LAMBDA_INT, config.LAMBDA_PHASE) == (0.5, 0.3, 0.2)
    assert config.DEFAULT_SEEDS == [42, 43, 44, 45, 46]
    assert config.DEFAULT_ABLATION_SEEDS == [42, 43, 44]
    assert (config.DEFAULT_EPOCHS, config.DEFAULT_ABLATION_EPOCHS) == (20, 15)


def test_va_prototypes_match_table3():
    expected = {  # Table 3 (VA columns)
        "angry": (-0.5, 0.7), "happy": (0.8, 0.7), "sad": (-0.7, 0.3),
        "neutral": (0.0, 0.5), "fear": (-0.6, 0.8), "disgust": (-0.6, 0.6),
        "surprise": (0.5, 0.9),
    }
    assert config.EMOTIONS == list(expected)
    for i, name in enumerate(config.EMOTIONS):
        np.testing.assert_allclose(config.VA_PROTOTYPES[i], expected[name], atol=1e-6)


@pytest.mark.parametrize("name,expected", [
    ("03-01-01-01-01-01-01.wav", 3),   # neutral
    ("03-01-02-01-01-01-01.wav", 3),   # calm merged into neutral
    ("03-01-03-01-01-01-01.wav", 1),   # happy
    ("03-01-04-01-01-01-01.wav", 2),   # sad
    ("03-01-05-01-01-01-01.wav", 0),   # angry
    ("03-01-06-01-01-01-01.wav", 4),   # fear
    ("03-01-07-01-01-01-01.wav", 5),   # disgust
    ("03-01-08-01-01-01-01.wav", 6),   # surprise
])
def test_ravdess_labels(name, expected):
    assert label_from_path(name, "ravdess") == expected


@pytest.mark.parametrize("name,expected", [
    ("OAF_back_angry.wav", 0), ("YAF_bar_happy.wav", 1), ("OAF_bite_sad.wav", 2),
    ("YAF_boat_neutral.wav", 3), ("OAF_book_fear.wav", 4), ("YAF_chain_disgust.wav", 5),
    ("OAF_dog_ps.wav", 6), ("YAF_door_pleasant_surprise.wav", 6),
])
def test_tess_labels(name, expected):
    assert label_from_path(name, "tess") == expected


def test_cremad_labels():
    assert label_from_path("1001_DFA_ANG_XX.wav".lower(), "cremad") == 0
    assert label_from_path("1001_IEO_NEU_XX.wav".lower(), "cremad") == 3


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        label_from_path("garbage.wav", "tess")

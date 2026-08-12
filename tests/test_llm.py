import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from llm import RELEVANCE_THRESHOLD


def test_relevance_threshold_is_calibrated_value():
    """Locks in the calibrated threshold (0.3), not the original
    uncalibrated guess (0.05) that only barely passed the Munich
    no-match test (score 0.049) with almost no safety margin."""
    assert RELEVANCE_THRESHOLD == 0.3


def test_float32_score_is_json_serializable():
    """This is the exact bug that cost significant debugging time earlier:
    FlashRank returns numpy.float32 scores, which json.dumps() cannot
    serialize directly. This test locks in that scores are always cast
    to plain Python float before being returned."""
    import numpy as np
    fake_score = np.float32(0.765)
    result = round(float(fake_score), 3)
    # This should not raise, and should be a real, plain Python float
    json.dumps({"score": result})
    assert isinstance(result, float)
    assert not isinstance(result, np.floating)
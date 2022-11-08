"""Utils unittest implementation."""

import numpy as np

from challenge.evaluator import Evaluator


def test_eval():
    """Test evaluation utility in challenge/evaluator.py."""
    t_pred_top10 = np.array(
        [
            [55, 38, 22, 39, 94, 91, 48, 59, 5, 66],
            [44, 35, 23, 36, 75, 92, 84, 39, 10, 58],
            [4, 9, 92, 38, 48, 58, 47, 24, 23, 19],
        ]
    )
    t = np.array([22, 11, 23])
    dir_path = 'temp'

    eval = Evaluator(t_pred_top10, t, dir_path)
    mrr = eval.eval_func()

    assert mrr > 0

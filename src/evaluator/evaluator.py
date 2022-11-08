"""Evaluator that returns mrr."""

from ogb.lsc import WikiKG90Mv2Evaluator

class Evaluator:
    """Performance evaluator for the competition.

    Args:
            t_pred_top10 (np.array or torch.Tensor): Top predictions that has shape of (num_test_triples, 10).
            t (np.array): True value that is used to rank t_pred_top10.
    """

    def __init__(self, t_pred_top10, t, dir_path):
        self.t_pred_top10 = t_pred_top10
        self.t = t
        self.dir_path = dir_path
        self.evaluator = WikiKG90Mv2Evaluator()

    def eval_func(self):
        """Evaulation function for predictions."""
        input_dict = {}
        input_dict['h,r->t'] = {'t_pred_top10': self.t_pred_top10, 't': self.t}
        result_dict = self.evaluator.eval(input_dict)

        return result_dict['mrr']  # get mrr

    def submit_result(self):
        """Submission for the competition."""
        input_dict = {}
        input_dict['h,r->t'] = {'t_pred_top10': self.t_pred_top10}
        # use 'test-challenge'for test submission
        self.evaluator.save_test_submission(input_dict=input_dict, dir_path=self.dir_path, mode='test-challenge')
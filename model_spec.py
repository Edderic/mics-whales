""" Integration tests for model """
from mamba import description, context, it, before
from lib.sampler import model_simple, model_quadratic_yspb

with description('model_simple') as self:
    with context('some values are set'):
        with before.each:
            self.test_params = {
                'age t-1': 1,
                'proba_alive t-1': 1,
                'proba_birth t-1': 0,
                'proba_observed t-1': 1,
                'alive_proba': 1,
                'unknown_birth_coeff': 1,
                'birth_intercept': 1,
                'proba_observed_given_alive': 1
            }

            self.subject = model_simple(self.test_params)['data']

        with it('should not have contiguous observed births'):
            for i in range(len(self.subject)-2):
                assert self.subject[i:i+2].sum() < 4

with description('model_quadratic_yspb') as self:
    with context('some values are set'):
        with before.each:
            self.test_params = {
                'age t-1': 1,
                'proba_alive t-1': 1,
                'proba_birth t-1': 0,
                'proba_observed t-1': 1,
                'alive_proba': 1,
                'yspb t-1': 8,
                'birth_intercept': 1,
                'birth_unknown': 1,
                'birth_peak_yspb': 1,
                'birth_width': -0.05,
                'proba_observed_given_alive': 1,
                'proba_had_a_birth_before': 1
            }

            self.subject = model_quadratic_yspb(self.test_params)['data']

        with it('should not have contiguous observed births'):
            for i in range(len(self.subject)-2):
                assert self.subject[i:i+2].sum() < 4

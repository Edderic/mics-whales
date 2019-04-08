""" Integration tests for model """
from mamba import description, context, it, before
from lib.sampler import model_simple

with description('model_simple') as self:
    with context('some values are set'):
        with before.each:
            self.num_years = 36
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
            for i in range(self.num_years - 2):
                assert self.subject[i:i+2].sum() < 4

        # with it('by default it should have 36 items'):

""" Integration tests for model """
from mamba import description, context, it, before
from lib.sampler import model

with description('model') as self:
    with context('some values are set'):
        with before.each:
            self.num_years = 36

            self.test_params = {
                'num_years': self.num_years,
                'start_age': 1,
                'start_alive': 1,
                'start_had_a_birth_before': 0,
                'start_yspb': 1,
                'had_no_births_yet_prior_constant': 1,
                'had_no_births_yet_prior_age': 1,
                'had_births_before_prior_constant': 1,
                'had_births_before_prior_age': 1,
                'had_births_before_yspb': 2,
                'had_births_before_prior_yspb': 1,
                'had_births_before_prior_yspb_squared': 1,
                'alive_proba': 0.95,
                'observed_count_prior_seen_t_minus_1': 1,
                'observed_count_prior_seen_before_t_minus_1': 1,
                'observed_count_prior_constant': 1,
            }

            self.subject = model(self.test_params)

        with it('should not have contiguous observed births'):
            for i in range(self.num_years - 2):
                assert self.subject[i:i+2].sum() < 4

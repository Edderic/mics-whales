""" Integration tests for model """
from mamba import description, context, it, before
from lib.sampler import model_quadratic_yspb

with description('model') as self:
    with context('some values are set'):
        with before.each:
            self.num_years = 36
            self.test_params = {
                'start_age': 1,
                'proba_start_alive': 1,
                'proba_start_had_a_birth_before': 0,
                'start_yspb': 1,
                'had_no_births_yet_prior_constant': 1,
                'had_no_births_yet_prior_age': 1,
                'had_births_before_prior_constant': 1,
                'had_births_before_prior_age': 1,
                'had_births_before_yspb': 2,
                'had_births_before_prior_peak_yspb': 1,
                'had_births_before_prior_width': 1,
                'alive_proba': 0.95,
                'observed_count_prior_seen_t_minus_1': 1,
                'observed_count_prior_seen_before_t_minus_1': 1,
                'observed_count_prior_constant': 1,
            }

            self.subject = model_quadratic_yspb(self.test_params)['data']

        with it('should not have contiguous observed births'):
            for i in range(self.num_years - 2):
                assert self.subject[i:i+2].sum() < 4

        # with it('by default it should have 36 items'):

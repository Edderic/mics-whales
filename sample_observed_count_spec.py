from mamba import description, context, it
from expects import expect, equal
from lib.sampler import sample_observed_count
import numpy as np

# TODO: improve brittleness (Check for range)
# Refactor duplication of arguments for instantiation

with description('sample_observed_count') as self:
    with before.each:
        self.alive_t=0
        self.birth_t=0
        self.observed_count_t_minus_1=0
        self.was_observed_t_minus_1=0
        self.prior_observed_count_t_minus_1=0
        self.prior_was_observed_t_minus_1=0
        self.constant=0

    # was_observed_t_minus_1 | observed_count_t_minus_1 |
        # 1                  |         > 0              |
        # 0                  |         > 0              |
        # 1                  |           0              |
        # 0                  |           0              | not seen at all (default)

    with description('when whale is not alive'):
        with before.each:
            self.alive_t = 0
        with it('should return 0'):
            sample = sample_observed_count(
                alive_t=self.alive_t,
                birth_t=self.birth_t,
                observed_count_t_minus_1=self.observed_count_t_minus_1,
                was_observed_t_minus_1=self.was_observed_t_minus_1,
                prior_observed_count_t_minus_1=self.prior_observed_count_t_minus_1,
                prior_was_observed_t_minus_1=self.prior_was_observed_t_minus_1,
                constant=self.constant,
            )

            assert sample == 0

    with description('when whale is alive'):
        with before.each:
            self.alive_t = 1

        with description('when whale was not observed'):
            with before.each:
                self.was_observed_t_minus_1 = 0

            with description('and whale did not show up last time'):
                with before.each:
                    self.observed_count_t_minus_1 = 0

                with description('and constant is very negative'):
                    with before.each:
                        self.constant = -15

                        self.sample = sample_observed_count(
                            alive_t=self.alive_t,
                            birth_t=self.birth_t,
                            observed_count_t_minus_1=self.observed_count_t_minus_1,
                            was_observed_t_minus_1=self.was_observed_t_minus_1,
                            prior_observed_count_t_minus_1=self.prior_observed_count_t_minus_1,
                            prior_was_observed_t_minus_1=self.prior_was_observed_t_minus_1,
                            constant=self.constant,
                        )

                    with it('should most likely give 0'):
                        assert self.sample == 0

        with description('when whale was observed'):
            with before.each:
                self.was_observed_t_minus_1 = 1

            with description('and the coefficient for that var is strong'):
                with before.each:
                    self.prior_was_observed_t_minus_1 = 100

                with description('and whale did not show up last time'):
                    with before.each:
                        self.observed_count_t_minus_1 = 0

                    with description('and there was a birth'):
                        with before.each:
                            self.birth_t = 1

                        with it('should return 2'):
                            self.sample = sample_observed_count(
                                alive_t=self.alive_t,
                                birth_t=self.birth_t,
                                observed_count_t_minus_1=self.observed_count_t_minus_1,
                                was_observed_t_minus_1=self.was_observed_t_minus_1,
                                prior_observed_count_t_minus_1=self.prior_observed_count_t_minus_1,
                                prior_was_observed_t_minus_1=self.prior_was_observed_t_minus_1,
                                constant=self.constant,
                            )

                            assert self.sample == 2


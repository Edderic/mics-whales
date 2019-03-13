""" This tests the sample_observed_count function """
from mamba import description, context, it, before
from lib.sampler import sample_observed_count

# TODO: improve brittleness (Check for range)
# Refactor duplication of arguments for instantiation

with description('sample_observed_count') as self:
    with before.each:
        self.args = {}
        self.args['alive_t'] = 0
        self.args['birth_t'] = 0
        self.args['seen_t_minus_1'] = 0
        self.args['was_observed_t_minus_1'] = 0
        self.args['prior_seen_t_minus_1'] = 0
        self.args['prior_was_observed_t_minus_1'] = 0
        self.args['constant'] = 0

    with context('when whale is not alive'):
        with before.each:
            self.args['alive_t'] = 0

        with it('should return 0'):
            sample = sample_observed_count(**self.args)

            assert sample == 0

    with context('when whale is alive'):
        with before.each:
            self.args['alive_t'] = 1

        with context('when whale was not observed'):
            with before.each:
                self.args['was_observed_t_minus_1'] = 0

            with context('and whale did not show up last time'):
                with before.each:
                    self.args['seen_t_minus_1'] = 0

                with context('and constant is very negative'):
                    with before.each:
                        self.args['constant'] = -15
                        self.sample = sample_observed_count(**self.args)

                    with it('should most likely give 0'):
                        assert self.sample == 0

        with context('when whale was observed'):
            with before.each:
                self.args['was_observed_t_minus_1'] = 1

            with context('and the coefficient for that var is strong'):
                with before.each:
                    self.args['prior_was_observed_t_minus_1'] = 100

                with context('and whale did not show up last time'):
                    with before.each:
                        self.args['seen_t_minus_1'] = 0

                    with context('and there was a birth'):
                        with before.each:
                            self.args['birth_t'] = 1

                        with it('should return 2'):
                            self.sample = sample_observed_count(**self.args)

                            assert self.sample == 2


""" This tests the sample_observed_count function """
from mamba import description, context, it, before
from lib.sampler import sample_observed_count
from spec.helpers import set_to_value_except

# TODO: improve brittleness (Check for range)
# Refactor duplication of arguments for instantiation

with description('sample_observed_count') as self:
    with before.each:
        self.args = {}
        self.keys_to_set_to = [
            'alive_t',
            'birth_t',
            'seen_previously',
            'seen_previously_coeff',
            'constant',
        ]

        set_to_value_except(
            self.args,
            self.keys_to_set_to,
            0,
            _except=[]
        )

    with context('when whale is not alive'):
        with before.each:
            self.args['alive_t'] = 0

        with it('should return 0'):
            sample = sample_observed_count(**self.args)

            assert sample == 0

    with context('when whale is alive'):
        with before.each:
            self.args['alive_t'] = 1

        with context('when whale was not observed previously'):
            with before.each:
                self.args['seen_previously'] = 0

            with context('and constant is very negative'):
                with before.each:
                    self.args['constant'] = -15
                    self.sample = sample_observed_count(**self.args)

                with it('should most likely give 0'):
                    assert self.sample == 0

        with context('when whale was observed previously'):
            with before.each:
                self.args['seen_previously'] = 1

            with context('and the coefficient for that var is strong'):
                with before.each:
                    self.args['seen_previously_coeff'] = 100

                with context('and there was a birth'):
                    with before.each:
                        self.args['birth_t'] = 1

                    with it('should return 2'):
                        self.sample = sample_observed_count(**self.args)

                        assert self.sample == 2


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
            'proba_observed_given_alive',
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

        with context('and proba_observed_given_alive is zero'):
            with before.each:
                self.args['proba_observed_given_alive'] = 0
                self.sample = sample_observed_count(**self.args)

            with it('will be 0'):
                assert self.sample == 0

        with context('and proba_observed_given_alive is 1'):
            with before.each:
                self.args['proba_observed_given_alive'] = 1

            with context('and there was a birth'):
                with before.each:
                    self.args['birth_t'] = 1

                with it('should return 2'):
                    self.sample = sample_observed_count(**self.args)

                    assert self.sample == 2


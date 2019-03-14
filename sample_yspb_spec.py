""" this tests sample_yspb """

from mamba import description, context, it, before
from lib.sampler import sample_yspb

with description('sample_yspb') as self:
    with context('when the whale has not had any births yet in previous years at all'):
        with before.each:
            self.args = {}
            self.args['had_a_birth_prior_to_t_minus_1'] = 0
            self.args['birth_t_minus_1'] = 0
            self.args['yspb_t_minus_1'] = 8 # this value is discarded

        with it('should raise an error'):
            try:
                sample_yspb(**self.args)
                assert False
            except ValueError as err:
                self.message = "sample_yspb only makes sense when there's been a birth before."
                assert err.args[0] == self.message

    with context('when the whale gave birth for the first time last year'):
        with before.each:
            self.args = {}
            self.args['had_a_birth_prior_to_t_minus_1'] = 0
            self.args['birth_t_minus_1'] = 1
            self.args['yspb_t_minus_1'] = 8 # this value is discarded

        with it('should return 1'):
            self.subject = sample_yspb(**self.args)
            assert self.subject == 1

    with context('when the whale had given birth previously before last year'):
        with before.each:
            self.args = {}
            self.args['had_a_birth_prior_to_t_minus_1'] = 1
            self.args['yspb_t_minus_1'] = 8 # this value is used

        with context('and no birth last year'):
            with before.each:
                self.args['birth_t_minus_1'] = 0

            with it('should increment yspb_t_minus_1'):
                self.subject = sample_yspb(**self.args)
                assert self.subject == (self.args['yspb_t_minus_1'] + 1)

        with context('and a birth last year'):
            with before.each:
                self.args['birth_t_minus_1'] = 1

            with it('should return 1'):
                self.subject = sample_yspb(**self.args)
                assert self.subject == 1

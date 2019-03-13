""" This tests the sample_seen_before_t function """
from mamba import description, context, it, before
from lib.sampler import sample_seen_before_t

with description('sample_seen_before_t') as self:
    with context('when whale was seen before'):
        with before.each:
            self.args = {}
            self.args['seen_before_t_minus_1'] = 1

        with context('when whale was not seen last year'):
            with before.each:
                self.args['seen_t_minus_1'] = 0

            with it('should return 1'):
                subject = sample_seen_before_t(**self.args)
                assert subject == 1

        with context('when whale was seen last year'):
            with before.each:
                self.args['seen_t_minus_1'] = 1

            with it('should still return 1'):
                subject = sample_seen_before_t(**self.args)
                assert subject == 1

    with context('when whale was NOT seen before'):
        with before.each:
            self.args = {}
            self.args['seen_before_t_minus_1'] = 0

        with context('when whale was not seen last year'):
            with before.each:
                self.args['seen_t_minus_1'] = 0

            with it('should return 0'):
                subject = sample_seen_before_t(**self.args)
                assert subject == 0

        with context('when whale was seen last year'):
            with before.each:
                self.args['seen_t_minus_1'] = 1

            with it('should still return 1'):
                subject = sample_seen_before_t(**self.args)
                assert subject == 1



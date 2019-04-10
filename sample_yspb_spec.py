""" this tests sample_yspb """

from mamba import description, context, it, before
from lib.sampler import sample_yspb

with description('sample_yspb') as self:
    with context('when the whale just gave birth'):
        with before.each:
            self.args = {}
            self.args['birth'] = 1
            self.args['yspb_year_before'] = 2

        with it('should give 0'):
            assert sample_yspb(**self.args) == 0

    with context('when the whale did not give birth'):
        with before.each:
            self.args = {}
            self.args['birth'] = 0
            self.args['yspb_year_before'] = 2

        with it('should increase the yspb from the year before'):
            assert sample_yspb(**self.args) == 3

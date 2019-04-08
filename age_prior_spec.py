""" Test for generating sensible priors for age for a certain year. """
from mamba import description, context, it, before
from lib.prior_generation import age_prior
import pandas as pd
from math import isclose

with description('age_prior') as self:
    with before.each:
        self.args = {}
        self.years = [str(i) for i in range(1980, 1986)]

    with context('when year is 1983'):
        with before.each:
            self.args['year'] = 1983

        with context('and whale was first seen in 1984'):
            with before.each:
                self.args['row'] = pd.Series([0, 0, 0, 0, 1, 0], index=self.years)

            with context('and whale was observed as a calf'):
                with before.each:
                    self.args['known'] = True

                with it('should give a uniform prior between -1 and 0 '):
                    self.rv = age_prior(**self.args)
                    assert self.rv.mean() == -0.5
                    assert isclose(self.rv.var(), 0.083, abs_tol=0.01)

            with context('and whale was NOT observed as a calf'):
                with before.each:
                    self.args['known'] = False

                with it('should give a uniform prior between 0 and 80 '):
                    self.rv = age_prior(**self.args)
                    assert self.rv.mean() == 40
                    assert isclose(self.rv.var(), 533.33, abs_tol=0.01)

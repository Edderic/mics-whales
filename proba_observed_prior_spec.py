"""
Test for generating sensible priors for having seen a whale in a given year.
"""
from mamba import description, context, it, before
from lib.prior_generation import proba_observed_year_prior
import pandas as pd

with description('proba_observed_year_prior') as self:
    with before.each:
        self.args = {}
        self.years = [str(i) for i in range(1980, 1986)]

    with context('when year is 1983'):
        with before.each:
            self.args['year'] = 1983

        with context('and row is [1, 0, 0, 0, 0, 0] starting at 1980'):
            with before.each:
                self.args['row'] = pd.Series([1, 0, 0, 0, 0, 0], index=self.years)

            with it('should give a strong prior suggesting whale was not seen'):
                self.rv = proba_observed_year_prior(**self.args)
                assert self.rv.mean() < 0.01
                assert self.rv.var() < 0.02

        with context('and row is [1, 0, 0, 1, 0, 0] starting at 1980'):
            with before.each:
                self.args['row'] = pd.Series([1, 0, 0, 1, 0, 0], index=self.years)

            with it('should give a strong prior suggesting whale was seen'):
                self.rv = proba_observed_year_prior(**self.args)
                assert self.rv.mean() > 0.99
                assert self.rv.var() < 0.02

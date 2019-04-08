""" Test for inferring if whale was alive or not at a certain year. """
from mamba import description, context, it, before
from lib.prior_generation import seen_previously_prior
import pandas as pd

with description('seen_previously_prior') as self:
    with before.each:
        self.args = {}
        self.years = [str(i) for i in range(1980, 1986)]

    with context('when year is 1983'):
        with before.each:
            self.args['year'] = 1983

        with context('and row is [1, 0, 0, 0, 0, 0] starting at 1980'):
            with before.each:
                self.args['row'] = pd.Series([1, 0, 0, 0, 0, 0], index=self.years)

            with it('should return a strong prior saying yes'):
                self.rv = seen_previously_prior(**self.args)
                assert self.rv.mean() > 0.99
                assert self.rv.var() < 0.02

        with context('and row is [0, 0, 0, 0, 1, 0] starting at 1980'):
            with before.each:
                self.args['row'] = pd.Series([0, 0, 0, 0, 1, 0], index=self.years)

            with it('should return a strong prior saying no'):
                self.rv = seen_previously_prior(**self.args)
                assert self.rv.mean() < 0.01
                assert self.rv.var() < 0.02

        with context('and whale was observed at the year 1983'):
            with context('but was not observed before'):
                with before.each:
                    self.args['row'] = pd.Series([0, 0, 0, 1, 0, 0], index=self.years)

                with it('should return a strong prior saying yes'):
                    self.rv = seen_previously_prior(**self.args)
                    assert self.rv.mean() > 0.99
                    assert self.rv.var() < 0.02

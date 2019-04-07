""" Test for inferring if whale gave birth on a certain year. """
from mamba import description, context, it, before
from lib.prior_generation import proba_birth_year_prior
import pandas as pd

with description('proba_birth_year_prior') as self:
    with before.each:
        self.args = {}
        self.years = [str(i) for i in range(1980, 1986)]

    with description('when year is 1983'):
        with before.each:
            self.args['year'] = 1983

        with context('when whale is observed to have given birth that year'):
            with before.each:
                self.args['row'] = pd.Series([1, 0, 0, 2, 0, 0], index=self.years)

            with it('should give a strong prior in favor of a birth that year'):
                self.rv = proba_birth_year_prior(**self.args)
                assert self.rv.mean() > 0.99 and self.rv.mean() < 1
                assert self.rv.var() < 0.02

        with context('when whale is observed NOT to have given birth that year'):
            with before.each:
                self.args['row'] = pd.Series([1, 0, 0, 1, 0, 0], index=self.years)

            with it('should give a strong prior in favor of not giving birth that year'):
                self.rv = proba_birth_year_prior(**self.args)
                assert self.rv.mean() < 0.01
                assert self.rv.var() < 0.02

        with context('when whale is NOT observed that year'):
            with context('but observed to have given birth the year after'):
                with before.each:
                    self.args['row'] = pd.Series([1, 0, 0, 0, 2, 0], index=self.years)

                with it('should give a strong prior in favor of not giving birth that year'):
                    self.rv = proba_birth_year_prior(**self.args)
                    assert self.rv.mean() < 0.01
                    assert self.rv.var() < 0.02

            with context('but observed to have given birth the year before'):
                with before.each:
                    self.args['row'] = pd.Series([1, 0, 2, 0, 0, 0], index=self.years)

                with it('should give a strong prior in favor of not giving birth that year'):
                    self.rv = proba_birth_year_prior(**self.args)
                    assert self.rv.mean() < 0.01
                    assert self.rv.var() < 0.02


        with context('when whale is NOT observed that year or the immediate years'):
            with context('but observed to have given birth the year after'):
                with before.each:
                    self.args['row'] = pd.Series([1, 0, 0, 0, 0, 0], index=self.years)

                with it('should give a weak prior'):
                    self.rv = proba_birth_year_prior(**self.args)
                    assert self.rv.mean() == 0.5
                    assert self.rv.var() > 0.083 and self.rv.var() < 0.084

""" Test for getting the minimum age of a whale """
from mamba import description, context, it, before
from lib.prior_generation import min_age_prior
import pandas as pd

with description('min_age_prior(row, year)') as self:
    with before.each:
        self.years = [str(i) for i in range(1980, 1986)]
        self.args = {}

    with context('when row is [1, 0, 1...] starting at 1980'):
        with before.each:
            self.args['row'] = pd.Series([1, 0, 1, 0, 0, 0], index=self.years)

        with context('when year is 1985'):
            with before.each:
                self.args['year'] = 1985

            with it('should give 5'):
                self.subject = min_age_prior(**self.args)
                assert self.subject == 5

    with context('when row is [0, 0, 1...] starting at 1980'):
        with before.each:
            self.args['row'] = pd.Series([0, 0, 1, 0, 0, 0], index=self.years)

        with context('when year is 1985'):
            with before.each:
                self.args['year'] = 1985

            with it('should give 3'):
                self.subject = min_age_prior(**self.args)
                assert self.subject == 3

    with context('when row is [0, 0, 1...] starting at 1980'):
        with before.each:
            self.years = [str(i) for i in range(1980, 1986)]
            self.args = {}
            self.args['row'] = pd.Series([0, 0, 1, 0, 0, 0], index=self.years)

        with context('when year is 1981'):
            with before.each:
                self.args['year'] = 1981

            with it('should give -1'):
                self.subject = min_age_prior(**self.args)
                assert self.subject == -1

    with context('when there are no sightings at all'):
        with before.each:
            self.args['row'] = pd.Series([0, 0, 0, 0, 0, 0], index=self.years)
            self.args['year'] = 1982

        with it('should raise an error'):
            try:
                min_age_prior(**self.args)
                assert False
            except RuntimeError as err:
                self.val = 1
                assert err.args[0] == 'Please make sure that ' + \
                        'the row has at least one sighting (i.e. value of 1 or 2)'

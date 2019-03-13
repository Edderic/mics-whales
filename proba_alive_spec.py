""" This tests proba_alive """
from mamba import description, context, it, before
from lib.sampler import proba_alive

with description('proba_alive') as self:
    with context('when model is geometric'):
        with before.each:
            self.subject_arguments = {
                'proba': 0.05
            }

        with context('when age is less than 0'):
            with before.each:
                self.subject_arguments['age'] = -1

            with context('when whale is alive the year before'):
                with before.each:
                    self.subject_arguments['alive_year_before'] = 0
                    self.subject = proba_alive(**self.subject_arguments)

                with it('should return 0'):
                    assert self.subject == 0

        with context('when age is zero'):
            with before.each:
                self.subject_arguments['age'] = 0

            with context('when whale was not alive the year before'):

                with before.each:
                    self.subject_arguments['alive_year_before'] = 1
                    self.subject = proba_alive(**self.subject_arguments)

                with it('should return 1'):
                    assert self.subject == 1

        with context('when age is greater than 0'):
            with before.each:
                self.subject_arguments['age'] = 1

            with context('when whale was alive the year before'):
                with before.each:
                    self.subject_arguments['alive_year_before'] = 1
                    self.subject = proba_alive(**self.subject_arguments)

                with it('should return the passed probability'):
                    assert self.subject == self.subject_arguments['proba']

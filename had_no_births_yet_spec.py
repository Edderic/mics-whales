""" Tests methods on HadNoBirthsYet """
from mamba import description, context, it, before
from lib.sampler import HadNoBirthsYet
from spec.helpers import set_to_value_except

with description('HadNoBirthsYet') as self:
    with before.each:
        self.args = {}
        self.arguments = [
            'age',
            'repr_active',
            'alive',
            'prior_constant',
            'prior_age'
        ]

    with description('#proba_give_birth'):
        with context('and everything else is set to 1'):
            with before.each:
                set_to_value_except(
                    args=self.args,
                    keys_to_set_to=self.arguments,
                    value=1,
                    _except=[]
                )


            with it('should return the weighted sum passed through logistic function'):
                self.subject = HadNoBirthsYet(**self.args).proba_give_birth()

                assert self.subject > 0.88 and self.subject < 0.89

            with context('and whale is not alive'):
                with before.each:
                    self.args['alive'] = 0
                    self.subject = HadNoBirthsYet(**self.args).proba_give_birth()

                with it('should return 0'):
                    assert self.subject == 0

            with context('and whale is alive'):
                with before.each:
                    self.args['alive'] = 1

                with context('but whale is not yet reproductively active'):
                    with before.each:
                        self.args['repr_active'] = 0
                        self.subject = HadNoBirthsYet(**self.args).proba_give_birth()

                    with it('should return 0'):
                        assert self.subject == 0

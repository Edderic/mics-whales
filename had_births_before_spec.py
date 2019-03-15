""" Tests methods on HadBirthsBefore """
from mamba import description, context, it, before
from lib.sampler import HadBirthsBefore
from spec.helpers import set_to_value_except

with description('HadBirthsBefore') as self:
    with before.each:
        self.args = {}
        self.arguments = [
            'age',
            'repr_active',
            'prior_constant',
            'prior_age',
            'yspb',
            'prior_yspb',
            'prior_yspb_squared'
        ]

    with description('#proba_give_birth'):
        with context('and everything is set to 1'):
            with before.each:
                set_to_value_except(
                    args=self.args,
                    keys_to_set_to=self.arguments,
                    value=1,
                    _except=[]
                )


            with context('but whale is NOT reproductively active'):
                with before.each:
                    self.args['repr_active'] = 0
                    self.subject = HadBirthsBefore(**self.args).proba_give_birth()

                with it('should return 0'):
                    assert self.subject == 0

            with context('and whale reproductively active'):
                with before.each:
                    self.args['repr_active'] = 1
                    self.subject = HadBirthsBefore(**self.args).proba_give_birth()

                with it('should return the weighted sums run through logistic function'):
                    assert self.subject > 0.98 and self.subject < 0.99

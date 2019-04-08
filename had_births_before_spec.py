""" Tests methods on HadBirthsBeforeQuadratic """
from mamba import description, context, it, before
from lib.sampler import HadBirthsBeforeQuadratic
from spec.helpers import set_to_value_except

with description('HadBirthsBeforeQuadratic') as self:
    with before.each:
        self.args = {}
        self.arguments = [
            'age',
            'repr_active',
            'prior_constant',
            'prior_age',
            'yspb',
            'prior_peak_yspb',
            'prior_width'
        ]

    with description('#proba_give_birth'):
        with context('and everything is set to 1 except yspb'):
            with before.each:
                set_to_value_except(
                    args=self.args,
                    keys_to_set_to=self.arguments,
                    value=1,
                    _except=['yspb']
                )

                self.args['yspb'] = 2

            with context('but whale is NOT reproductively active'):
                with before.each:
                    self.args['repr_active'] = 0
                    self.subject = HadBirthsBeforeQuadratic(**self.args).proba_give_birth()

                with it('should return 0'):
                    assert self.subject == 0

            with context('and whale reproductively active'):
                with before.each:
                    self.args['repr_active'] = 1
                    self.subject = HadBirthsBeforeQuadratic(**self.args).proba_give_birth()

                with it('should return the weighted sums run through logistic function'):
                    assert self.subject > 0.95 and self.subject < 0.96

        with context('when whale gave birth last year'):
            with before.each:
                set_to_value_except(
                    args=self.args,
                    keys_to_set_to=self.arguments,
                    value=1,
                    _except=[]
                )
                self.subject = HadBirthsBeforeQuadratic(**self.args).proba_give_birth()

            with it('should return 0'):
                assert self.subject == 0

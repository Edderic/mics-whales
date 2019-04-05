""" Tests proba_birth. """
from mamba import description, context, it, before
from lib.sampler import proba_birth, logistic
from spec.helpers import set_to_value_except

with description('proba-birth') as self:
    with before.each:
        self.args = {}
        self.keys_to_set_to = [
            'repr_active',
            'birth_last_year',
            'age',
            'age_coeff',
            'intercept'
        ]

    with context('whale is not reproductively active'):
        with before.each:
            set_to_value_except(
                self.args,
                self.keys_to_set_to,
                value=1,
                _except=['repr_active']
            )

            self.args['repr_active'] = 0

            self.subject = proba_birth(**self.args)

        with it('should not give birth'):
            assert self.subject == 0

    with context('whale is reproductively active'):
        with before.each:
            self.args['repr_active'] = 1

        with context('but gave birth last year'):
            with before.each:
                self.args['birth_last_year'] = 1

                set_to_value_except(
                    self.args,
                    self.keys_to_set_to,
                    value=0,
                    _except=['repr_active', 'birth_last_year']
                )

                self.subject = proba_birth(**self.args)

            with it('should return 0'):
                assert self.subject == 0

        with context('and did not give birth last year'):
            with before.each:
                self.args['birth_last_year'] = 0

            with context('and everything has else set to 1'):
                with before.each:

                    set_to_value_except(
                        self.args,
                        self.keys_to_set_to,
                        value=1,
                        _except=['repr_active', 'birth_last_year']
                    )

                    self.subject = proba_birth(**self.args)

                with it('should give a proba about 0.88'):
                    assert self.subject > 0.88 and self.subject < 0.89

from mamba import description, context, it
from expects import expect, equal
from lib.sampler import sample_birth
import numpy as np



with description('Sample Birth Spec') as self:
    with description('when whale is not alive'):
        with before.each:
            self.alive_t = 0

        with it('should have a mean of zero'):
            collection = []

            for i in range(100):
                sample = sample_birth(
                    repr_active_t=1,
                    alive_t=self.alive_t,
                    unobs_t_minus_1=0,
                    unobs_t_minus_1_on_birth_t=0,
                    unobs_t_minus_1_on_birth_t__no_births=0,
                    age_t=0,
                    age_t_on_birth_t=0,
                    age_t_on_birth_t__no_births=0,
                    constant_on_birth_t=0,
                    constant_on_birth_t__no_births=0,
                    yspb_t=0,
                    yspb_t_on_birth_t=0,
                    yspb_t_squared_on_birth_t=0,
                )

                collection.append(sample)

            mean = np.array(collection).mean()

            expect(mean).to(equal(0))

    with description('when whale is not reproductively active'):
        with before.each:
            self.repr_active_t = 0

        with it('should give a mean of 1'):
            collection = []

            for i in range(100):
                sample = sample_birth(
                    repr_active_t=self.repr_active_t,
                    alive_t=1,
                    unobs_t_minus_1=0,
                    unobs_t_minus_1_on_birth_t=0,
                    unobs_t_minus_1_on_birth_t__no_births=0,
                    age_t=0,
                    age_t_on_birth_t=0,
                    age_t_on_birth_t__no_births=0,
                    constant_on_birth_t=0,
                    constant_on_birth_t__no_births=0,
                    yspb_t=0,
                    yspb_t_on_birth_t=0,
                    yspb_t_squared_on_birth_t=0,
                )

                collection.append(sample)

            mean = np.array(collection).mean()

            expect(mean).to(equal(0))

    with description('when whale just gave birth last year'):
        with before.each:
            self.yspb_t = 1

        with it("can't give birth this year"):
            collection = []

            for i in range(100):
                sample = sample_birth(
                    repr_active_t=1,
                    alive_t=1,
                    unobs_t_minus_1=0,
                    unobs_t_minus_1_on_birth_t=0,
                    unobs_t_minus_1_on_birth_t__no_births=0,
                    age_t=0,
                    age_t_on_birth_t=0,
                    age_t_on_birth_t__no_births=0,
                    constant_on_birth_t=0,
                    constant_on_birth_t__no_births=0,
                    yspb_t=self.yspb_t,
                    yspb_t_on_birth_t=0,
                    yspb_t_squared_on_birth_t=0,
                )

                collection.append(sample)

            mean = np.array(collection).mean()

            expect(mean).to(equal(0))

    with description('when whale is reproductively active and alive'):
        with before.each:
            self.repr_active_t = 1
            self.alive_t = 1

        with description('when whale has not given birth at all'):
            with before.each:
                self.yspb_t = -1

            with description('and age is a super strong predictor'):
                with before.each:
                    self.age_t_on_birth_t__no_births = 1

                with description('and whale has high age'):
                    with before.each:
                        self.age_t = 50

                    with it('should on average produce a birth'):

                        collection = []

                        for i in range(100):
                            sample = sample_birth(
                                repr_active_t=self.repr_active_t,
                                alive_t=self.alive_t,
                                unobs_t_minus_1=0,
                                unobs_t_minus_1_on_birth_t=0,
                                unobs_t_minus_1_on_birth_t__no_births=0,
                                age_t=self.age_t,
                                age_t_on_birth_t=0,
                                age_t_on_birth_t__no_births=1,
                                constant_on_birth_t=0,
                                constant_on_birth_t__no_births=0,
                                yspb_t=self.yspb_t,
                                yspb_t_on_birth_t=0,
                                yspb_t_squared_on_birth_t=0,
                            )

                            collection.append(sample)

                        mean = np.array(collection).mean()

                        assert mean > 0.9

            with description('and age is NOT a predictor at all'):
                with before.each:
                    self.age_t_on_birth_t__no_births = 0
                    self.age_t = 50

                with description('but the constant is a strong predictor'):
                    with before.each:
                        self.constant_on_birth_t__no_births = 100

                    with it('should on average produce a birth'):
                        collection = []

                        for i in range(100):
                            sample = sample_birth(
                                repr_active_t=self.repr_active_t,
                                alive_t=self.alive_t,
                                unobs_t_minus_1=0,
                                unobs_t_minus_1_on_birth_t=0,
                                unobs_t_minus_1_on_birth_t__no_births=0,
                                age_t=self.age_t,
                                age_t_on_birth_t=0,
                                age_t_on_birth_t__no_births=1,
                                constant_on_birth_t=0,
                                constant_on_birth_t__no_births=self.constant_on_birth_t__no_births,
                                yspb_t=self.yspb_t,
                                yspb_t_on_birth_t=0,
                                yspb_t_squared_on_birth_t=0,
                            )

                            collection.append(sample)

                        mean = np.array(collection).mean()

                        assert mean > 0.9

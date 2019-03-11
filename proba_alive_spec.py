from mamba import description, context, it
from expects import expect, equal
from lib.sampler import proba_alive
import pandas as pd

with description('proba_alive') as self:
    with description('and priors are set to ones'):
        with before.each:
            self.prior_age = 1
            self.prior_constant = 1

        with description('and we are interested in row H002 and year 1985'):
            with before.each:
                self.whale_id = 'H002'
                self.year = '1985'

            with description('when whale was seen before and seen afterwards'):
                with before.each:
                    self.df = pd.DataFrame(
                        [
                            [1, 0, 0, 1],
                            [0, 0, 0, 0]
                        ],
                        columns=['1983', '1984', '1985', '1986'],
                        index=['H002', 'H003']
                    )

                    self.age = 1
                    self.prior_age = 1
                    self.prior_constant = 1

                    self.subject = proba_alive(
                        age=self.age,
                        prior_age=self.prior_age,
                        prior_constant=self.prior_constant,
                        whale_id=self.whale_id,
                        year=self.year,
                        df=self.df
                    )

                with it('should return 1.0'):
                    assert self.subject == 1.0

            with description('when whale was ONLY seen before AND not seen afterwards'):
                with before.each:
                    self.df = pd.DataFrame(
                        [
                            [1, 0, 0, 0],
                            [0, 0, 0, 0]
                        ],
                        columns=['1983', '1984', '1985', '1986'],
                        index=['H002', 'H003']
                    )

                with description('when age is less than 0'):
                    with before.each:
                        self.age = -1

                        self.subject = proba_alive(
                            age=self.age,
                            prior_age=self.prior_age,
                            prior_constant=self.prior_constant,
                            whale_id=self.whale_id,
                            year=self.year,
                            df=self.df
                        )

                    with it('should return 0.0'):
                        assert self.subject == 0.0


                with description('when age is greater than 0'):
                    with before.each:
                        self.age = 1

                        self.subject = proba_alive(
                            age=self.age,
                            prior_age=self.prior_age,
                            prior_constant=self.prior_constant,
                            whale_id=self.whale_id,
                            year=self.year,
                            df=self.df
                        )

                    with it('should return xyz'):
                        assert self.subject > 0.88 and self.subject < 0.89

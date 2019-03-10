from mamba import description, context, it
from expects import expect, equal
from lib.sampler import from_start_year_up_to_final_year
import pandas as pd

with description('from_start_year_up_to_final_year'):
    with before.each:
        self.df = pd.DataFrame(
            [
                [1,2,3,4,5],
                [6,7,8,9,10]
            ],
            columns=['1989', '1990', '1991', '1992', '1993'],
            index=['H002', 'H003']
        )
        self.whale_id = 'H002'
        self.start_year = '1991'

        self.subject = from_start_year_up_to_final_year(
            df=self.df,
            whale_id=self.whale_id,
            start_year=self.start_year
        )

    with it('should have 3 items'):
        assert len(self.subject) == 3

    with it('should have 3 on the first item'):
        assert self.subject.iloc[0] == 3

    with it('should have 4 on the first item'):
        assert self.subject.iloc[1] == 4

    with it('should have 5 on the first item'):
        assert self.subject.iloc[2] == 5

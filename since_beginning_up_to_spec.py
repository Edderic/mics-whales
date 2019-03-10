from mamba import description, context, it
from expects import expect, equal
from lib.sampler import since_beginning_up_to
import pandas as pd

with description('since_beginning_up_to') as self:
    with description('when columns are years since 1979 and indices are H002 & H003'):
        with before.each:
            self.columns = ['1979', '1980', '1981', '1982', '1983', '1984']
            self.rows = ['H002', 'H003']

        with description('when a row is [0, 1, 2, 0, 1, 2]'):
            with before.each:
                self.df = pd.DataFrame(
                    [
                        [0, 1, 2, 3, 4, 5],
                        [0, 0, 0, 0, 0, 0]
                    ],
                    columns=self.columns,
                    index=self.rows
                )

            with description('and we are interested in data up to 1982 for H002'):
                with before.each:
                    self.subject = since_beginning_up_to(
                        df=self.df,
                        row_index='H002',
                        up_to='1982'
                    )

                with it('should have the first item be 0'):
                    assert self.subject.loc['1979'] == 0

                with it('should have the second item be 1'):
                    assert self.subject.loc['1980'] == 1

                with it('should have the second item be 2'):
                    assert self.subject.loc['1981'] == 2

                with it('should have the second item be 3'):
                    assert self.subject.loc['1982'] == 3

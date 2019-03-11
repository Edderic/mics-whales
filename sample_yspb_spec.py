from mamba import description, context, it
from expects import expect, equal
from lib.sampler import plausible_yspb
import pandas as pd


with description('plausible_yspb') as self:
    with description('when df has years as columns and indexed by whale id'):
        with before.each:
            self.columns = [
                '1979', '1980', '1981', '1982', '1983', '1984',
                '1985', '1986', '1987', '1988', '1989'
            ]
            self.rows = ['H002', 'H003']

        with description('and the data for H002 is [0,1,1,0,1,0,1,2,0,0,0]'):
            with before.each:
                self.first_row = [0, 1, 1, 0, 1, 0, 1, 2, 0, 0, 0]
                self.second_row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            with description('and the up to year was 1988'):
                with before.each:
                    self.up_to_year = '1988'
                    self.dataframe = pd.DataFrame(
                        [
                            self.first_row,
                            self.second_row
                            ],
                        columns=self.columns,
                        index=self.rows
                    )

                with description('when age is 15'):
                    with before.each:
                        self.age = 15

                        self.subject = plausible_yspb(
                            row_index='H002',
                            age=self.age,
                            up_to_year=self.up_to_year,
                            df=self.dataframe
                        )

                    with it('should say that H002 could have given birth 3 years ago'):
                        assert 3 in self.subject

                    with it('should NOT say that H002 could have given birth 2 years ago'):
                        assert 2 not in self.subject

                    with it('should say that H002 could have given birth 1 year ago'):
                        assert 1 in self.subject

        with description('and the data for H002 is [0,1,1,0,1,0,1,2,0,1,0]'):
            with before.each:
                self.first_row = [0, 1, 1, 0, 1, 0, 1, 2, 0, 1, 0]
                self.second_row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            with description('and the up to year was 1988'):
                with before.each:
                    self.up_to_year = '1988'
                    self.dataframe = pd.DataFrame(
                        [
                            self.first_row,
                            self.second_row
                            ],
                        columns=self.columns,
                        index=self.rows
                    )

                with description('when age is 15'):
                    with before.each:
                        self.age = 15

                        self.subject = plausible_yspb(
                            row_index='H002',
                            age=self.age,
                            up_to_year=self.up_to_year,
                            df=self.dataframe
                        )

                    with it('should say that H002 could have given birth 3 years ago'):
                        assert 3 in self.subject

                    with it('should NOT say that H002 could have given birth 2 years ago'):
                        assert 2 not in self.subject

        with description('and the data for H002 is [0,1,1,0,1,0,1,0,1,1,0]'):
            with before.each:
                self.first_row = [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
                self.second_row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            with description('and the up to year was 1988'):
                with before.each:
                    self.up_to_year = '1988'
                    self.dataframe = pd.DataFrame(
                        [
                            self.first_row,
                            self.second_row
                            ],
                        columns=self.columns,
                        index=self.rows
                    )

                with description('when age is 15'):
                    with before.each:
                        self.age = 15

                        self.subject = plausible_yspb(
                            row_index='H002',
                            age=self.age,
                            up_to_year=self.up_to_year,
                            df=self.dataframe
                        )

                    with it('should say that H002 could have given birth 3 years ago'):
                        assert 3 in self.subject

                    with it('should say that H002 could have given birth 5 years ago'):
                        assert 5 in self.subject

                    with it('should say that H002 could NOT have given birth 7 years ago'):
                        assert 7 not in self.subject

                    with it('should say that H002 could NOT have given birth 10 years ago'):
                        assert 10 not in self.subject

                    with it('should only have 2 potential answers'):
                        assert len(self.subject) == 2

                with description('when age is 21'):
                    with before.each:
                        self.age = 21

                        self.subject = plausible_yspb(
                            row_index='H002',
                            age=self.age,
                            up_to_year=self.up_to_year,
                            df=self.dataframe
                        )

                    with it('should say that H002 could have given birth 3 years ago'):
                        assert 3 in self.subject

                    with it('should say that H002 could have given birth 5 years ago'):
                        assert 5 in self.subject

                    with it('should say that H002 could have given birth 7 years ago'):
                        assert 7 in self.subject

                    with it('should say that H002 could have given birth 10 years ago'):
                        assert 10 in self.subject

                    with it('should say that H002 could have given birth 11 years ago'):
                        assert 11 in self.subject

                    with it('should only have 5 possible answers'):
                        assert len(self.subject) == 5

                with description('when age is 8'):
                    with before.each:
                        self.age = 8

                        self.subject = plausible_yspb(
                            row_index='H002',
                            age=self.age,
                            up_to_year=self.up_to_year,
                            df=self.dataframe
                        )

                    with it('should return an empty list'):
                        assert len(self.subject) == 0



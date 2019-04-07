"""
    This module is meant to house functions that assist in generating sensible
    priors.
"""

import numpy as np
from pyabc import RV

BIRTH_EVENT = 2
MOM_ONLY_EVENT = 1

def min_age_prior(row, year):
    """
        Get the minimum possible age of the whale relative to the given year.

        Parameters:
            row: pd.Series
                the row that represents data for one female.

            year: integer
                The year we'd like to know the minimum age of the whale

        Returns: integer. Minimum age of the whale
    """

    if row.sum() == 0:
        raise RuntimeError('Please make sure that the row has at least one ' + \
                'sighting (i.e. value of 1 or 2)')
    sightings = np.where(row >= 1)[0]

    first_sighting_year = int(row.index[sightings[0]])

    return year - first_sighting_year

def proba_alive_year_prior(row, year):
    """
        Probability of being alive for a certain year, taking into account
        whether the whale was sighted later on.
    """
    final_year = int(row.index[-1])
    years = [str(i) for i in range(year, final_year + 1)]

    if row.loc[years].sum() == 0:
        return RV('uniform', 0, 1)
    else:
        return RV('beta', 100, 1)

def proba_birth_year_prior(row, year):
    """
        Probability of having given birth for a certain year.
    """

    prev_year = year - 1
    next_year = year + 1

    if row.loc[str(year)] == BIRTH_EVENT:
        return RV('beta', 100, 1)
    elif row.loc[str(year)] == MOM_ONLY_EVENT:
        return RV('beta', 1, 100)
    elif row.loc[str(prev_year)] == BIRTH_EVENT:
        return RV('beta', 1, 100)
    elif row.loc[str(next_year)] == BIRTH_EVENT:
        return RV('beta', 1, 100)
    else:
        return RV('uniform', 0, 1)


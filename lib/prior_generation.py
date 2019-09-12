"""
    This module is meant to house functions that assist in generating sensible
    priors.
"""

import numpy as np

BIRTH_EVENT = 2
MOM_ONLY_EVENT = 1
UNOBSERVED_EVENT = 0
MAX_AGE = 80 # years

# TODO: might be better to just make this into an age_prior, to be consistent
# with other methods?

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

    if row.sum() == UNOBSERVED_EVENT:
        raise RuntimeError('Please make sure that the row has at least one ' + \
                'sighting (i.e. value of 1 or 2)')
    sightings = np.where(row >= MOM_ONLY_EVENT)[0]

    first_sighting_year = int(row.index[sightings[0]])

    return year - first_sighting_year
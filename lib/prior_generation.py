"""
    This module is meant to house functions that assist in generating sensible
    priors.
"""

import numpy as np
from pyabc import RV

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

def age_prior(row, year, known=True):
    """
        Generates sensible priors for age at a given year.

        Parameters:
            row: pd.Series
                the row that represents data for one female.

            year: integer
                The year we'd like to generate a prior for.

            known: boolean
                If known, the age prior is limited to a certain range.

        Returns: RV.
    """
    lower_age = min_age_prior(row, year)

    if known:
        return RV('uniform', lower_age, 1)
    else:
        # we assume the whale was not born the year we first saw the whale
        # The whale, however, could have been born the year before.
        upper_age = MAX_AGE - (lower_age + 1)
        return RV('uniform', lower_age + 1, upper_age)

def proba_alive_prior(row, year):
    """
        Probability of being alive for a certain year, taking into account
        whether the whale was sighted later on.

        Parameters:
            row: pd.Series
                the row that represents data for one female.

            year: integer
                The year we'd like to know the minimum age of the whale

        Returns: RV.
    """
    final_year = int(row.index[-1])
    years = [str(i) for i in range(year, final_year + 1)]

    if row.loc[years].sum() == UNOBSERVED_EVENT:
        return RV('uniform', 0, 1)
    else:
        return RV('beta', 100, 1)

def proba_birth_prior(row, year):
    """
        Probability of having given birth for a certain year.

        Parameters:
            row: pd.Series
                the row that represents data for one female.

            year: integer
                The year we'd like to know the probability of birth.

        Returns: RV.
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

def proba_observed_prior(row, year):
    """
        Probability of having observed the whale for a certain year.

        Parameters:
            row: pd.Series
                the row that represents data for one female.

            year: integer
                The year whose observation we're interested in.

        Returns: RV.
    """

    if row.loc[str(year)] == UNOBSERVED_EVENT:
        return RV('beta', 1, 100)
    else:
        return RV('beta', 100, 1)

def seen_previously_prior(row, year):
    """
        Probability of having observed the whale for a certain year and the
        years previous to that.

        Parameters:
            row: pd.Series
                the row that represents data for one female.

            year: integer
                The year whose observation we're interested in.

        Returns: RV.
    """

    earliest_year = int(row.index[0])
    years = [str(i) for i in range(earliest_year, year + 1)]

    if row.loc[years].sum() == 0:
        return RV('beta', 1, 100)
    else:
        return RV('beta', 100, 1)

def generate_priors_for_individual(row, start_year, known):
    """
    """
    row.index[row.index]
    # known =
    year_before_start_year = start_year - 1

    # notes:
    # alive_proba:  favors higher values, but lower values still
    #   plausible
    #
    # observed_count_seen_previously_coeff: being seen before means
    #   that the whale was alive at some point in the past. If the
    #   whale was seen before, we might be more likely to see the whale
    #   at some point later on.

    return {
        'age t-1': age_prior(row, year_before_start_year, known=known),
        'proba_alive t-1': proba_alive_prior(row, year_before_start_year),
        'proba_birth t-1': proba_alive_prior(row, year_before_start_year),
        'proba_observed t-1': proba_observed_prior(row, year_before_start_year),
        'alive_proba': RV('beta', 2.5, 1.0),
        'unknown_birth_coeff': RV('norm', 0, 0.75),
        'birth_intercept': RV('norm', 0, 2),
        'proba_observed_given_alive': RV('beta', 2.5, 1.0),
    }


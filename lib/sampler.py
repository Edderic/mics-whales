"""
    This module contains functions for sampling.
"""
import numpy as np

NO_BIRTHS_YET = None

def logistic(val):
    """
        This is the sigmoid function. Takes a real number and converts
        it into a number between 0 and 1.
    """
    return 1.0 / (1.0 + np.exp(-val))

def proba_birth(age):
    """
        What is the probability of giving birth?
    """
    return 1

def sample_birth(
    repr_active_t,
    alive_t,
    unobs_t_minus_1,
    unobs_t_minus_1_on_birth_t,
    unobs_t_minus_1_on_birth_t__no_births,
    age_t,
    age_t_on_birth_t,
    age_t_on_birth_t__no_births,
    constant_on_birth_t,
    constant_on_birth_t__no_births,
    yspb_t,
    yspb_t_on_birth_t,
    yspb_t_squared_on_birth_t,
):
    if alive_t == 0:
        return 0
    if repr_active_t == 0:
        return 0
    if yspb_t == 1:
        return 0

    if yspb_t == NO_BIRTHS_YET:
        p_birth_t = logistic(
            age_t * age_t_on_birth_t__no_births + \
            unobs_t_minus_1 * unobs_t_minus_1_on_birth_t__no_births + \
            constant_on_birth_t__no_births
        )

    return np.random.binomial(n=1, p=p_birth_t)


def sample_observed_count(
    alive_t,
    birth_t,
    seen_t_minus_1,
    was_observed_t_minus_1,
    prior_seen_t_minus_1,
    prior_was_observed_t_minus_1,
    constant,
):
    """
        Simulates the observed count.

        Parameters:
            alive_t: boolean. Was whale alive at time t?
            birth_t: boolean. Did whale give birth at time t?

            seen_t_minus_1: integer. What was the observed count
                in the year prior

            was_observed_t_minus_1: prior to the calculation of
                seen_t_minus_1, was the whale observed?

            prior_seen_t_minus_1: The prior for the GLM

            constant: The prior for the GLM in the case that whale was
                not observed at all

        Returns: 0 (unobserved), 1 (observed, no birth), or
            2 (observed w/ birth)

        Notes:

            Three potential cases:

            | wsb t-1 | obs t-1 | equation                                 |
            |    0    |    0    | c
            |    1    |    1    | c + obs(t-1)*prior('obs', t-1) + \
                                     wo(t-1)*prior('wo', t-1)              |
            |    1    |    0    | c + wo(t-1)*prior('wo', t-1)             |
            |    0    |    1    | doesn't happen                           |

            c: constant
            obs(t-1): observed in t-1
            wo(t-1): was observed in t-1
    """

    if alive_t == 0:
        return 0

    p = logistic(
        seen_t_minus_1 * prior_seen_t_minus_1 + \
        was_observed_t_minus_1 * prior_was_observed_t_minus_1 + constant
    )

    if np.random.binomial(n=1, p=p) == 1:
        return alive_t + birth_t
    else:
        return 0

def since_beginning_up_to(
    df,
    row_index,
    up_to
):
    """
        Gives a row for an individual whale with columns starting from the
        very beginning up to the specified year.

        E.g. Let's assume df is the following
                1980, 1981, 1982, 1983
            H01   1     2     3     4
            H02   5     6     7     8

        since_beginning_up_to(df=df, row_index='H01', up_to='1982')
        should return
                1980, 1981, 1982,
            H01   1     2     3

        Parameters:
            df: DataFrame indexed by whale ids (strings),
                columns are years (strings).

            row_index: string. The whale id.

            up_to: string. A year in the form "XXXX" (e.g. "1982")

        Returns:
            A row
    """

    row = df.loc[row_index]
    up_to_index = np.where(df.columns == up_to)[0][0] + 1

    return row.iloc[0:up_to_index]

def plausible_yspb(
    row_index,
    age,
    df,
    up_to_year
):
    """
        Gives plausible values for Years Since Previous Birth.

        Parameters:
            row_index: String. The id of the whale, based on the
                dataframe passed in.

            age: integer. Could be positive, zero, or negative.

            df: DataFrame. Index are whale ids. Columns are years (e.g. '1980')

            up_to_year: String. A year (e.g. '1982')

        Returns: an array. Could be empty.
    """

    row_values = since_beginning_up_to(
        df=df,
        row_index=row_index,
        up_to=up_to_year
    )

    UNSPOTTED = 0
    SPOTTED_FEMALE_AND_CALF = 2
    MAX_YEARS = 50
    REPRO_AGE = 9

    index_of_current_year = row_values.shape[0]

    indices_of_known_give_birth = np.where(
        row_values == SPOTTED_FEMALE_AND_CALF
    )[0]

    indices_of_maybe_give_birth = np.where(row_values == UNSPOTTED)[0]

    collection = []

    if len(indices_of_known_give_birth) > 0:
        latest_known_give_birth_year = indices_of_known_give_birth.max()

        collection.append(latest_known_give_birth_year)

        for x in indices_of_maybe_give_birth:
            if x > latest_known_give_birth_year + 1:
                collection.append(x)
    else:
        collection = list(np.arange(-MAX_YEARS,0))

        for x in indices_of_maybe_give_birth:
            collection.append(x)


    yspbs = [index_of_current_year - x for x in collection]
    plausible_years_since_previous_births = \
            [x for x in yspbs if age - x > REPRO_AGE]

    return plausible_years_since_previous_births

def sample_yspb(
    row_index,
    age,
    df,
    up_to_year
):
    """
        Samples plausible values for Years Since Previous Birth.

        Parameters:
            row_index: String. The id of the whale, based on the
                dataframe passed in.

            age: integer. Could be positive, zero, or negative.

            df: DataFrame. Index are whale ids. Columns are years (e.g. '1980')

            up_to_year: String. A year (e.g. '1982')

        Returns: a value. None means whale hasn't given birth yet.
    """

    potential_yspbs = plausible_yspb(
        row_index,
        age,
        df,
        up_to_year
    )

def proba_alive(
    age,
    alive_year_before,
    proba
):
    """
        Gives the probability of the whale being alive at a certain year.

        Parameters:
            age: integer. could be negative, zero, or positive

            alive_year_before: integer. 0 for false, 1 for true.

            proba: The probability of being alive assuming whale was alive
                the year before

        Returns: a probability
    """
    if age < 0:
        return 0.0
    elif age == 0:
        return 1.0 # we assume the whale would be alive at least for the first year
    elif alive_year_before == 0:
        return 0.0
    else:
        return proba

def sample_alive(
    age,
    alive_year_before,
    proba
):
    """
        Simulates being alive.

        Parameters:
            age: integer. could be negative, zero, or positive

            alive_year_before: integer. 0 for false, 1 for true.

            proba: The probability of being alive assuming whale was alive
                the year before

        Returns: 0 (dead) or 1 (alive)
    """

    proba = proba_alive(
        age,
        alive_year_before,
        proba
    )

    return np.random.binomial(n=1, p=proba)

def from_start_year_up_to_final_year(
    df,
    start_year,
    whale_id
):

    """
        Gives a row for an individual whale with columns starting from
        start_year up to the final year.

        E.g. Let's assume df is the following
                1980, 1981, 1982, 1983
            H01   1     2     3     4
            H02   5     6     7     8

        from_start_year_up_to_final_year(df=df, whale_id='H01', start_year='1982')
        should return
                1982, 1983
            H01   3     4

        Parameters:
            df: DataFrame indexed by whale ids (strings),
                columns are years (strings).

            whale_id: string. The whale id.

            start_year: string. A year in the form "XXXX" (e.g. "1982")

        Returns:
            A row (pandas Series).
    """

    row = df.loc[whale_id]
    start_index = np.where(df.columns == start_year)[0][0]

    return row.iloc[start_index:]

def sample_seen_before_t(
    seen_before_t_minus_1,
    seen_t_minus_1
):
    """
        Was the whale seen before for time t?

        Parameters:

            seen_before_t_minus_1: boolean. Was whale seen at time t-1 or before?

            seen_t_minus_1: boolean. Was whale seen at time t-1?

        Returns: boolean.
    """
    if seen_before_t_minus_1 == 1 or seen_t_minus_1 == 1:
        return 1
    else:
        return 0

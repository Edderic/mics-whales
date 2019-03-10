import numpy as np

def logistic(val):
    return 1.0 / (1.0 + np.exp(-val))

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

    not_born_yet = -1

    if yspb_t == not_born_yet:
        p_birth_t = logistic(
            age_t * age_t_on_birth_t__no_births + \
            unobs_t_minus_1 * unobs_t_minus_1_on_birth_t__no_births + \
            constant_on_birth_t__no_births
        )

    return np.random.binomial(n=1, p=p_birth_t)


def sample_observed_count(
    alive_t,
    birth_t,
    observed_count_t_minus_1,
    was_observed_t_minus_1,
    prior_observed_count_t_minus_1,
    prior_was_observed_t_minus_1,
    constant,
):
    """
        Simulates the observed count.

        Parameters:
            alive_t: boolean. Was whale alive at time t?
            birth_t: boolean. Did whale give birth at time t?

            observed_count_t_minus_1: integer. What was the observed count
                in the year prior

            was_observed_t_minus_1: prior to the calculation of
                observed_count_t_minus_1, was the whale observed?

            prior_observed_count_t_minus_1: The prior for the GLM

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
        observed_count_t_minus_1 * prior_observed_count_t_minus_1 + \
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

    row = df.loc[row_index]
    up_to_index = np.where(df.columns == up_to)[0][0] + 1

    return row.iloc[0:up_to_index]

def plausible_yspb(
    row_index,
    age,
    df,
    up_to_year
):

    row_values = since_beginning_up_to(
        df=df,
        row_index=row_index,
        up_to=up_to_year
    )

    UNSPOTTED = 0
    SPOTTED_FEMALE_AND_CALF = 2
    MAX_YEARS = 50

    index_of_current_year = row_values.shape[0]

    indices_of_known_give_birth = np.where(
        row_values == SPOTTED_FEMALE_AND_CALF
    )[0]

    indices_of_maybe_give_birth = np.where(row_values == UNSPOTTED)[0]

    collection = []

    if len(indices_of_known_give_birth) > 0:
        latest_known_give_birth_year = indices_of_known_give_birth.max()

        for x in indices_of_maybe_give_birth:
            if x > latest_known_give_birth_year:
                collection.append(x)
    else:
        collection = list(np.arange(-MAX_YEARS,0))

        for x in indices_of_maybe_give_birth:
            collection.append(x)


    yspbs = [index_of_current_year - x for x in collection]
    plausible_years_since_previous_births = [x for x in yspbs if age - x > 9]

    return plausible_years_since_previous_births


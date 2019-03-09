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


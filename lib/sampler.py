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

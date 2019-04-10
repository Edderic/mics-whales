"""
    This module contains functions for sampling.
"""
import numpy as np
import pandas as pd

def model_quadratic_yspb(parameters, num_years=11):
    """
        Produces data for an individual.

        parameters: a dictionary
            age t-1: integer (pos/zero/neg). The age of the whale in the year previous to
                the first year of predictables.

            proba_alive t-1: Probability of whale being alive the year before the
                predictables' first year.

            proba_birth t-1: Probability of whale having given birth
                the year before the predictables' first year?

            proba_observed t-1: Probability of having seen the whale
                the year before the predictables' first year?

            alive_proba: If whale was alive the year before, what's the probability
                of being alive now?

            yspb t-1: Years Since Previous Birth w.r.t. the year before the
                predictables' first year.

            birth_intercept: if whale hasn't given birth, how likely is the whale
                to give birth?

            birth_unknown: effect of unknown variable on probability of birth.

            birth_peak_yspb: The most likely YSPB value.

            birth_width: Determines the width of the parabola, and whether the
                "bowl" is up or down.

            proba_observed_given_alive: If alive, what's the probability of
                being observed?

            proba_had_a_birth_before: Probability of having given birth
                before the predictables.

        returns: a dictionary with 'data' attribute


    """
    ages = [parameters['age t-1']]
    alive = [np.random.binomial(n=1, p=parameters['proba_alive t-1'])]
    births = [np.random.binomial(n=1, p=parameters['proba_birth t-1'])]
    yspb = [parameters['yspb t-1']]
    observed = [np.random.binomial(n=1, p=parameters['proba_observed t-1'])]
    repr_actives = [sample_repr_active(age=ages[0], alive=alive[0])]
    had_a_birth_before = [np.random.binomial(n=1, p=parameters['proba_had_a_birth_before'])]

    for i in range(1, num_years):
        ages.append(ages[i-1] + 1)

        alive.append(
            sample_alive(
                age=ages[i],
                alive_year_before=alive[i-1],
                proba=parameters['alive_proba']
            )
        )

        repr_actives.append(sample_repr_active(age=ages[i], alive=alive[i]))

        proba_give_birth = proba_birth_quadratic_yspb(
            birth_last_year=births[i-1],
            birth_unknown=parameters['birth_unknown'],
            birth_intercept=parameters['birth_intercept'],
            birth_width=parameters['birth_width'],
            birth_peak_yspb=parameters['birth_peak_yspb'],
            unknown_value=i,
            yspb_year_before=yspb[i-1],
            had_a_birth_before=had_a_birth_before[i-1]
        )

        births.append(np.random.binomial(n=1, p=proba_give_birth))

        yspb.append(sample_yspb(birth=births[i], yspb_year_before=yspb[i-1]))

        had_a_birth_before.append(int(had_a_birth_before[i-1] == 1 or births[i] == 1))

        observed.append(
            sample_observed_count(
                alive_t=alive[i],
                birth_t=births[i],
                proba_observed_given_alive=parameters['proba_observed_given_alive']
            )
        )

    return {
        'data': np.array(observed[1:]),
        'debug': pd.DataFrame(
            {
                'observed_counts': observed,
                'alive': alive,
                'yspb': yspb,
                'repr_actives': repr_actives,
                'had_a_birth_before': had_a_birth_before,
            }
        )
    }


def proba_birth_quadratic_yspb(
        birth_last_year,
        birth_unknown,
        birth_intercept,
        birth_width,
        birth_peak_yspb,
        unknown_value,
        yspb_year_before,
        had_a_birth_before
):
    """
        Gives the probability of birth, taking into account Years Since
        Previous Birth (YSPB), and assuming a quadratic relationship.
    """
    if birth_last_year:
        return 0

    return logistic(
        had_a_birth_before * birth_width \
                * (yspb_year_before - birth_peak_yspb) ** 2
        + unknown_value * birth_unknown + birth_intercept
    )
def sample_repr_active(age, alive):
    """
        Samples reproductively activeness

        parameters:
            age: integer

        returns 1 or 0
    """
    if age > 9 and alive:
        return 1
    else:
        return 0

def logistic(val):
    """
        This is the sigmoid function. Takes a real number and converts
        it into a number between 0 and 1.
    """

    return 1.0 / (1.0 + np.exp(-val))

def sample_observed_count(
        alive_t,
        birth_t,
        proba_observed_given_alive,
):
    """
        Simulates the observed count.

        Parameters:
            alive: boolean. Was whale alive?
            birth: boolean. Did whale give birth?

            proba_observed_given_alive: The prior for the GLM in the case that whale
                is alive

        Returns: 0 (unobserved), 1 (observed, no birth), or
            2 (observed w/ birth)
    """

    if alive_t == 0:
        return 0

    if np.random.binomial(n=1, p=proba_observed_given_alive) == 1:
        return alive_t + birth_t
    else:
        return 0

def sample_yspb(
        birth,
        yspb_year_before,
):
    """
        Samples plausible values for Years Since Previous Birth.

        Parameters:
            birth: 0 or 1. Did whale give birth in the year of interest?

            yspb_year_before: 0 or 1. What was the years since previous birth
                in the year before the year of interest?

        Returns: an integer (1 or greater).
    """

    if birth:
        return 0
    else:
        return yspb_year_before + 1

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

def proba_birth(
        repr_active,
        birth_last_year,
        intercept,
        unknown,
        unknown_coeff=0,
):
    """
        Gives a probability of birth for some year of interest.

        Parameters:

            repr_active: 0 or 1. Is whale reproductively active?

            birth_last_year: 0 or 1. Did whale give birth last year?

            unknown: float. >= 0. What's the age this year of the whale?

            unknown_coeff: The effect of the unobserved on giving birth


        Returns: float. A probability.
    """
    if repr_active == 0 or birth_last_year == 1:
        return 0.0

    return logistic(unknown_coeff * unknown + intercept)

def model_simple(parameters, num_years=11):
    """
        Produces data for an individual.

        parameters: a dictionary
            num_years: positive integer. Number of years to produce data for

            age t-1: integer (pos/zero/neg). The age of the whale in the year previous to
                the first year of predictables.

            proba_alive t-1: Probability of whale being alive the year before the
                predictables' first year.

            proba_birth t-1: Probability of whale having given birth
                the year before the predictables' first year?

            proba_observed t-1: Probability of having seen the whale
                the year before the predictables' first year?

            alive_proba: If whale was alive the year before, what's the probability
                of being alive now?

            unknown_birth_coeff: regression coefficient for unknown var to predict birth

            birth_intercept: regression coeff for intercept on predicting birth

            proba_observed_given_alive: The prior for the GLM in the case that
                whale is alive.

        returns: a dictionary with 'data' attribute.
            data: list.
                The first item is the first year to be predicted. The length should equal
                num_years.


    """
    # first item is for the previous year
    ages = [parameters['age t-1']]
    alive = [np.random.binomial(n=1, p=parameters['proba_alive t-1'])]
    births = [np.random.binomial(n=1, p=parameters['proba_birth t-1'])]
    observed_count = [np.random.binomial(n=1, p=parameters['proba_observed t-1'])]
    repr_actives = [
        sample_repr_active(age=ages[0], alive=alive[0])
    ]

    for i in range(1, num_years+1):
        ages.append(ages[i-1] + 1)

        alive.append(
            sample_alive(
                age=ages[i],
                alive_year_before=alive[i-1],
                proba=parameters['alive_proba']
            )
        )

        repr_actives.append(
            sample_repr_active(age=ages[i], alive=alive[i])
        )

        _proba_birth = proba_birth(
            repr_active=repr_actives[i],
            birth_last_year=births[i-1],
            unknown=i,
            unknown_coeff=parameters['unknown_birth_coeff'],
            intercept=parameters['birth_intercept']
        )

        births.append(
            np.random.binomial(
                n=1,
                p=_proba_birth
            )
        )

        observed_count.append(
            sample_observed_count(
                alive_t=alive[i],
                birth_t=births[i],
                proba_observed_given_alive=parameters['proba_observed_given_alive'],
            )
        )

    return {
        'data': np.array(observed_count[1:]),
        'debug': pd.DataFrame(
            {
                'observed_counts_w_t-1': observed_count,
                'alive_w_t-1': alive,
                'repr_actives_w_t-1': repr_actives,
                'births_w_t-1': births,
            }
        )
    }

"""
    This module contains functions for sampling.
"""
import numpy as np
import pandas as pd

def model(parameters):
    """
        Produces data for an individual.

        parameters: a dictionary
            num_years: positive integer. Number of years to produce data for

            start_age: integer (pos/zero/neg). The age of the whale in the beginning.

            start_alive: Was whale alive in the beginning?

            start_had_a_birth_before: Did the whale have a birth before the
                time window of interest?

            start_yspb: years since previous birth.

            had_no_births_yet_prior_constant: constant term in the logistic
                regression for proba_give_birth.

            had_no_births_yet_prior_age: coefficient for age in the logistic
                regression for proba_give_birth.

            had_births_before_prior_constant: intercept term for logistic
                regression for HadBirthsBefore#proba_give_birth

            had_births_before_prior_age: age coefficient for logistic
                regression for HadBirthsBefore#proba_give_birth

            had_births_before_prior_yspb: coefficient for yspb in the logistic
                regression for proba_give_birth.

            had_births_before_prior_yspb_squared: coefficient for yspb**2 in the
                logistic regression for proba_give_birth.

            alive_proba: If whale was alive the year before, what's the probability
                of being alive now?

            observed_count_prior_seen_t_minus_1: The prior for the GLM

            observed_count_prior_seen_before_t_minus_1: The prior for the GLM in the case that
                whale was not observed at all

            observed_count_prior_constant: The prior for the GLM in the case that
                whale was not observed at all

        returns: a dictionary with 'data' attribute


    """
    ages = np.zeros(parameters['num_years'])
    ages[0] = parameters['start_age']

    alive = np.zeros(parameters['num_years'])
    alive[0] = parameters['start_alive']

    had_a_birth_before = np.zeros(parameters['num_years'])
    had_a_birth_before[0] = parameters['start_had_a_birth_before']

    births = np.zeros(parameters['num_years'])

    yspb = np.zeros(parameters['num_years'])
    yspb[0] = parameters['start_yspb']

    observed_count = np.zeros(parameters['num_years'])

    repr_actives = np.zeros(parameters['num_years'])
    seen_before_t_minus_1 = np.zeros(parameters['num_years'])
    seen_t_minus_1 = np.zeros(parameters['num_years'])

    for i in range(1, parameters['num_years']):
        ages[i] = ages[i-1] + 1

        alive[i] = sample_alive(
            age=ages[i],
            alive_year_before=alive[i-1],
            proba=parameters['alive_proba']
        )

        repr_actives[i] = sample_repr_active(age=ages[i], alive=alive[i])

        birth_proba_generator = None

        if had_a_birth_before[i-1] == 1:
            # TODO: check validity
            yspb[i] = sample_yspb(
                had_a_birth_prior_to_t_minus_1=int(had_a_birth_before[:i].sum() > 0),
                birth_t_minus_1=births[i-1],
                yspb_t_minus_1=yspb[i-1]
            )

            birth_proba_generator = HadBirthsBefore(
                age=ages[i],
                repr_active=repr_actives[i],
                prior_constant=parameters['had_births_before_prior_yspb'],
                prior_age=parameters['had_births_before_prior_age'],
                yspb=yspb[i],
                prior_yspb=parameters['had_births_before_prior_yspb'],
                prior_yspb_squared=parameters['had_births_before_prior_yspb_squared']
            )
        else:
            birth_proba_generator = HadNoBirthsYet(
                age=ages[i],
                repr_active=repr_actives[i],
                prior_constant=parameters['had_no_births_yet_prior_constant'],
                prior_age=parameters['had_no_births_yet_prior_age'],
            )

        proba_give_birth = birth_proba_generator.proba_give_birth()
        births[i] = np.random.binomial(n=1, p=proba_give_birth)

        had_a_birth_before[i] = births[i] or had_a_birth_before[i-1]

        seen_t_minus_1[i] = int(observed_count[i-1] > 0)
        seen_before_t_minus_1[i] = int(observed_count[:i].sum() > 0)

        observed_count[i] = sample_observed_count(
            alive_t=alive[i],
            birth_t=births[i],
            seen_t_minus_1=seen_t_minus_1[i],
            seen_before_t_minus_1=seen_before_t_minus_1[i],
            prior_seen_t_minus_1=parameters['observed_count_prior_seen_t_minus_1'],
            prior_seen_before_t_minus_1=parameters['observed_count_prior_seen_before_t_minus_1'],
            constant=parameters['observed_count_prior_constant']
        )

    return {
        'data': observed_count,
        'debug': pd.DataFrame(
            {
                'observed_counts': observed_count,
                'alive': alive,
                'yspb': yspb,
                'repr_actives': repr_actives,
                'had_a_birth_before': had_a_birth_before,
                'seen_t_minus_1': seen_t_minus_1,
                'seen_before_t_minus_1': seen_before_t_minus_1
            }
        )
    }


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

class HadNoBirthsYet:
    """
        Class that calculates the probability of giving birth,
        specifically for whales that hadn't given birth yet

        parameters:
            age: integer. The age of the whale.

            repr_active: 0 or 1. Is whale reproductively active?

            prior_constant: constant term in the logistic regression.

            prior_age: coefficient for age in the logistic regression.
    """
    def __init__(
            self,
            age,
            repr_active,
            prior_constant,
            prior_age
    ):
        self.age = age
        self.repr_active = repr_active
        self.prior_constant = prior_constant
        self.prior_age = prior_age


    def proba_give_birth(self):
        """
            Returns a probability of giving birth, between 0 and 1.
        """
        if self.repr_active == 0:
            return 0

        summation = self.age * self.prior_age + self.prior_constant
        return logistic(summation)

class HadBirthsBefore:
    """
        Class that calculates the probability of giving birth,
        specifically for whales that had given birth yet

        parameters:
            age: integer. The age of the whale.

            repr_active: 0 or 1. Is whale reproductively active?

            prior_constant: constant term in the logistic regression for proba_give_birth.

            prior_age: coefficient for age in the logistic regression for proba_give_birth.

            yspb: years since previous birth.

            prior_yspb: coefficient for yspb in the logistic regression for proba_give_birth.

            prior_yspb_squared: coefficient for yspb**2 in the logistic regression for proba_give_birth.

    """
    def __init__(
            self,
            age,
            repr_active,
            prior_constant,
            prior_age,
            yspb,
            prior_yspb,
            prior_yspb_squared
    ):
        self.age = age
        self.repr_active = repr_active
        self.prior_constant = prior_constant
        self.prior_age = prior_age
        self.yspb = yspb
        self.prior_yspb = prior_yspb
        self.prior_yspb_squared = prior_yspb_squared


    def proba_give_birth(self):
        """
            Returns a probability of giving birth, between 0 and 1.
        """
        if self.repr_active == 0 or self.yspb == 1:
            return 0

        summation = self.age * self.prior_age + \
                self.prior_constant + self.yspb * self.prior_yspb + \
                self.yspb ** 2 * self.prior_yspb_squared

        return logistic(summation)

def logistic(val):
    """
        This is the sigmoid function. Takes a real number and converts
        it into a number between 0 and 1.
    """

    return 1.0 / (1.0 + np.exp(-val))

def sample_observed_count(
        alive_t,
        birth_t,
        seen_t_minus_1,
        seen_before_t_minus_1,
        prior_seen_t_minus_1,
        prior_seen_before_t_minus_1,
        constant,
):
    """
        Simulates the observed count.

        Parameters:
            alive_t: boolean. Was whale alive at time t?
            birth_t: boolean. Did whale give birth at time t?

            seen_t_minus_1: integer. What was the observed count
                in the year prior

            seen_before_t_minus_1: prior to the calculation of
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

    proba = logistic(
        seen_t_minus_1 * prior_seen_t_minus_1 + \
        seen_before_t_minus_1 * prior_seen_before_t_minus_1 + constant
    )

    if np.random.binomial(n=1, p=proba) == 1:
        return alive_t + birth_t
    else:
        return 0

def sample_yspb(
        had_a_birth_prior_to_t_minus_1,
        birth_t_minus_1,
        yspb_t_minus_1
):
    """
        Samples plausible values for Years Since Previous Birth.

        Parameters:
            had_a_birth_prior_to_t_minus_1: integer (0 or 1). Besides
                birth_t_minus_1, did the whale have a birth prior to that?

            birth_t_minus_1: integer (0 or 1). Did the whale give birth last year?

            yspb_t_minus_1: integer. Number of years since previous birth (1 or greater).

        Returns: an integer (1 or greater). Could raise a ValueError if
            there hasn't been any births yet at all.
    """

    if had_a_birth_prior_to_t_minus_1 == 0 and birth_t_minus_1 == 0:
        raise ValueError("sample_yspb only makes sense when there's been a birth before.")
    if had_a_birth_prior_to_t_minus_1 == 1 and birth_t_minus_1 == 0:
        return yspb_t_minus_1 + 1
    if birth_t_minus_1 == 1:
        return 1

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

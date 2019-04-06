"""
    This module contains functions for sampling.
"""
import numpy as np
import pandas as pd

def model_quadratic_yspb(parameters, num_years=37):
    """
        Produces data for an individual.

        parameters: a dictionary
            num_years: positive integer. Number of years to produce data for

            start_age: integer (pos/zero/neg). The age of the whale in the beginning.

            proba_start_alive: Probability of whale being alive in the beginning.

            start_had_a_birth_before: Did the whale have a birth before the
                time window of interest?

            start_yspb: years since previous birth.

            had_no_births_yet_prior_constant: constant term in the logistic
                regression for proba_give_birth.

            had_no_births_yet_prior_age: coefficient for age in the logistic
                regression for proba_give_birth.

            had_births_before_prior_constant: intercept term for logistic
                regression for HadBirthsBeforeQuadratic#proba_give_birth

            had_births_before_prior_age: age coefficient for logistic
                regression for HadBirthsBeforeQuadratic#proba_give_birth

            had_births_before_prior_peak_yspb: The most likely YSPB value.

            had_births_before_prior_width: Determines the width of the parabola,
                and whether the "bowl" is up or down.

            alive_proba: If whale was alive the year before, what's the probability
                of being alive now?

            observed_count_prior_seen_t_minus_1: The prior for the GLM

            observed_count_prior_seen_before_t_minus_1: The prior for the GLM in the case that
                whale was not observed at all

            observed_count_prior_constant: The prior for the GLM in the case that
                whale was not observed at all

        returns: a dictionary with 'data' attribute


    """
    ages = np.zeros(num_years)
    ages[0] = parameters['start_age']

    alive = np.zeros(num_years)
    alive[0] = np.random.binomial(n=1, p=parameters['proba_start_alive'])

    had_a_birth_before = np.zeros(num_years)
    had_a_birth_before[0] = np.random.binomial(n=1, p=parameters['proba_start_had_a_birth_before'])

    births = np.zeros(num_years)

    yspb = np.zeros(num_years)
    yspb[0] = parameters['start_yspb']

    observed_count = np.zeros(num_years)

    repr_actives = np.zeros(num_years)
    seen_before_t_minus_1 = np.zeros(num_years)
    seen_t_minus_1 = np.zeros(num_years)

    for i in range(1, num_years):
        ages[i] = ages[i-1] + 1

        alive[i] = sample_alive(
            age=ages[i],
            alive_year_before=alive[i-1],
            proba=parameters['alive_proba']
        )

        repr_actives[i] = sample_repr_active(age=ages[i], alive=alive[i])

        birth_proba_generator = None

        if had_a_birth_before[i-1] == 1:
            yspb[i] = sample_yspb(
                had_a_birth_prior_to_t_minus_1=int(had_a_birth_before[:i].sum() > 0),
                birth_t_minus_1=births[i-1],
                yspb_t_minus_1=yspb[i-1]
            )

            birth_proba_generator = HadBirthsBeforeQuadratic(
                age=ages[i],
                repr_active=repr_actives[i],
                prior_constant=parameters['had_births_before_prior_constant'],
                prior_age=parameters['had_births_before_prior_age'],
                yspb=yspb[i],
                prior_peak_yspb=parameters['had_births_before_prior_peak_yspb'],
                prior_width=parameters['had_births_before_prior_width']
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

class HadBirthsBeforeQuadratic:
    """
        Class that calculates the probability of giving birth,
        specifically for whales that had given birth yet

        parameters:
            age: integer. The age of the whale.

            repr_active: 0 or 1. Is whale reproductively active?

            prior_constant: constant term in the logistic regression for proba_give_birth.

            prior_age: coefficient for age in the logistic regression for proba_give_birth.

            yspb: years since previous birth.

            prior_peak_yspb: float. coefficient for yspb in the logistic regression
                for proba_give_birth.

            prior_width: float. Influences how "wide" the parabola is, and whether or
                not it's upside down

    """
    def __init__(
            self,
            age,
            repr_active,
            prior_constant,
            prior_age,
            yspb,
            prior_peak_yspb,
            prior_width
    ):
        self.age = age
        self.repr_active = repr_active
        self.prior_constant = prior_constant
        self.prior_age = prior_age
        self.yspb = yspb
        self.prior_peak_yspb = prior_peak_yspb
        self.prior_width = prior_width

    def proba_give_birth(self):
        """
            Returns a probability of giving birth, between 0 and 1.
        """
        if self.repr_active == 0 or self.yspb == 1:
            return 0

        summation = self.age * self.prior_age + \
                self.prior_constant + \
                self.prior_width * (self.yspb - self.prior_peak_yspb) ** 2

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
        seen_previously,
        seen_previously_coeff,
        constant,
        unknown=0,
        unknown_coeff=0
):
    """
        Simulates the observed count.

        Parameters:
            alive: boolean. Was whale alive?
            birth: boolean. Did whale give birth?

            seen_previously: prior to the calculation of
                seen_t_minus_1, was the whale observed?

            seen_previously_coeff: The prior for the GLM

            constant: The prior for the GLM in the case that whale was
                not observed at all

        Returns: 0 (unobserved), 1 (observed, no birth), or
            2 (observed w/ birth)
    """

    if alive_t == 0:
        return 0

    proba = logistic(
        seen_previously * seen_previously_coeff + \
        unknown * unknown_coeff + constant
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

            seen_previously t-1: Was the whale observed before?

            alive_proba: If whale was alive the year before, what's the probability
                of being alive now?

            unknown_birth_coeff: regression coefficient for unknown var to predict birth

            birth_intercept: regression coeff for intercept on predicting birth

            observed_count_seen_previously_coeff: The prior for the GLM

            observed_count_constant: The prior for the GLM in the case that
                whale was not observed at all

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
    seen_previously = [parameters['seen_previously t-1']]
    repr_actives = [0] # placeholder

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
                seen_previously=seen_previously[i-1],
                seen_previously_coeff=parameters['observed_count_seen_previously_coeff'],
                constant=parameters['observed_count_constant'],
                unknown=0,
                unknown_coeff=0
            )
        )

        seen_previously.append(
            int(
                (seen_previously[i-1] == 1) or
                (observed_count[i] == 1)
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
                'seen_previously_w_t-1': seen_previously,
            }
        )
    }

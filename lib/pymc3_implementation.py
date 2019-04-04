"""

This module has functions to help us infer birth rate of reproductively active
females using PyMC3.

"""

import pymc3 as pm

def alive(
        age,
        alive_year_before,
        proba,
        name,
        zero,
        one,
        observed=None
):
    """
        PyMC3 sample alive
    """
    proba_alive = pm.math.switch(
        pm.math.lt(age, zero),
        zero,
        pm.math.switch(
            pm.math.eq(age, zero),
            one,
            pm.math.switch(
                pm.math.eq(alive_year_before, zero),
                zero,
                proba
            )
        )
    )

    return pm.Binomial(name, n=1, p=proba_alive, observed=observed)

def repr_active(age, alive, age_cutoff, zero, one):
    """
        Is the whale reproductively active?
    """

    return pm.math.switch(
        pm.math.gt(age, age_cutoff),
        pm.math.switch(
            pm.math.gt(alive, zero),
            one,
            zero
        ),
        zero
    )

def birth_with_no_yspb(
        repr_active,
        prior_constant,
        name,
        zero,
        observed=None
):
    """
        parameters:
            repr_active: 0 or 1. Is whale reproductively active?

            prior_constant: constant term in the logistic regression.

            prior_repr_active: coefficient for repr_active in the logistic regression.
    """

    proba =  pm.math.switch(
        pm.math.eq(repr_active, zero),
        zero,
        pm.math.invlogit(prior_constant)
    )

    return pm.Binomial(name, n=1, p=proba, observed=observed)

def birth_with_yspb_quadratic(
        age,
        repr_active,
        prior_constant,
        prior_age,
        yspb,
        prior_peak_yspb,
        prior_width,
        zero,
        one,
        name,
        observed=None
):

    """
        Returns a probability of giving birth, between 0 and 1.

        parameters:
            age: integer. The age of the whale.

            repr_active: 0 or 1. Is whale reproductively active?

            prior_constant: constant term in the logistic regression for proba_give_birth.

            prior_age: coefficient for age in the logistic regression for proba_give_birth.

            yspb: years since previous birth.

            prior_peak_yspb: float. coefficient for yspb in the logistic regression for proba_give_birth.

            prior_width: float. Influences how "wide" the parabola is, and whether or not it's upside down
    """
    proba = pm.math.switch(
        pm.math.or_(
            pm.math.eq(repr_active, yspb),
            pm.math.eq(yspb, one)
        ),
        zero,
        pm.math.invlogit(
          age * prior_age + \
            prior_constant + \
            prior_width * (yspb - prior_peak_yspb) ** 2
        )
    )

    return pm.Binomial(name, n=1, p=proba, observed=observed)

def observed_count(
        alive_t,
        birth_t,
        seen_before_t_minus_1,
        prior_seen_before_t_minus_1,
        constant,
        name,
        observed=None
):
    """
        Gives the probability of observing the count

        Parameters:
            alive_t: boolean. Was whale alive at time t?
            birth_t: boolean. Did whale give birth at time t?

            seen_before_t_minus_1: prior to the calculation of
                seen_t_minus_1, was the whale observed?

            prior_seen_before_t_minus_1: The prior for the GLM
                that captures the effect of being seen before

            constant: The prior for the GLM in the case that whale was
                not observed at all

        Returns: 0 (unobserved), 1 (observed, no birth), or
            2 (observed w/ birth)

    """

    proba = pm.math.invlogit(
        alive_t * seen_before_t_minus_1 * prior_seen_before_t_minus_1 + \
        constant
    )

    return pm.Binomial(name, n=1, p=proba, observed=observed)

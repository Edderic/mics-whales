""" Helper functions for specs """

def set_to_value_except(args, keys_to_set_to, value, _except):
    """
        Helper function to set values of arguments to the function at hand.

        Parameters:
            args: dictionary. Arguments to be passed to a function.

            keys_to_set_to: the keys in args whose value we set to the value
                argument.

            value: the value we set the arguments to

            _except: a list. A list of keys to exempt.
    """

    to_set_to_1 = list(set(keys_to_set_to) - set(_except))

    for i in to_set_to_1:
        args[i] = value

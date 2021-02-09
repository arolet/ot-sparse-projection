import sys


def get_logger(verbose, indent_level=0, period=1):
    """ A factory for a log function object

    This function allows to set a log level format once and for all without having to use if statements everywhere.

    Args:
        verbose: an int or boolean which defines whether to actually log stuff of not
        indent_level: an int that defines how many tabs to add as a prefix to log strings
        period: if set and above 1, this int will allow to ignore all calls to log(s, args, i) where i is not a multiple
        of period

    Returns:
        A log function, which takes as an input
    """
    if not verbose:
        return nothing
    prefix = "\t" * indent_level
    if period > 1:
        def logger(string, arguments=None, i=0):
            if i % period == 0:
                if arguments is not None:
                    s = "{}{}".format(prefix, string.format(*arguments))
                else:
                    s = "{}{}".format(prefix, string)
                print(s)
                sys.stdout.flush()

        return logger

    def logger(string, arguments=None, i=0):
        if arguments is not None:
            s = "{}{}".format(prefix, string.format(*arguments))
        else:
            s = "{}{}".format(prefix, string)

        print(s)
        sys.stdout.flush()

    return logger


def nothing(*args, **kwargs):
    """A utility function which does nothing"""
    pass

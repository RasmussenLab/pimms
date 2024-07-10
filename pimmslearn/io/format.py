def class_full_module(cls):
    """Return entire class name (repr notation) as str."""
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module not in ["__builtin__", "builtins"]:
        name = module + "." + name
    return name


def classname(obj):
    """
    Return entire object's class name (repr notation) as str.
    Source: https://gist.github.com/clbarnes/edd28ea32010eb159b34b075687bb49e

    Parameters
    ----------
    obj : object
        any object

    Returns
    -------
    str
        Full class name with module name
    """
    cls = type(obj)
    return class_full_module(cls)


class bcolors:
    """
    Class for colors changing string represenations in output.

    Found: https://stackoverflow.com/a/287944/9684872

    There are more options available:
    """

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

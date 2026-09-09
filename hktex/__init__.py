__modules__ = {}
__version__ = "0.1.0"


def register(name):
    def decorator(cls):
        if name in __modules__:
            raise ValueError(
                f"Module {name} already exists! Names of extensions conflict!"
            )
        __modules__[name] = cls
        return cls

    return decorator


def find(name):
    return __modules__[name]


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("hktex")


def is_debug():
    return logger.isEnabledFor(logging.DEBUG)


def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


def info(*args, **kwargs):
    logger.info(*args, **kwargs)


def warn(*args, **kwargs):
    logger.warning(*args, **kwargs)


from . import data, modules, trainers

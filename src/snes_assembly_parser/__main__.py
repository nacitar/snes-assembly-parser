from sevaht_utility.log_utility import log_exceptions

from .application import main

if __name__ == "__main__":
    raise SystemExit(log_exceptions()(main)())

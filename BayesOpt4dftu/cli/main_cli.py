import argparse

from BayesOpt4dftu import __package_name__, __version__


def parse_args():
    parser = argparse.ArgumentParser(description=f'{__package_name__} CLI. Use "./input.json" as the config file.')
    parser.add_argument('--version', action='version', version=f"{__package_name__} {__version__}")
    return parser.parse_args()

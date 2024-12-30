import argparse

from BayesOpt4dftu import __package_name__, __version__
from BayesOpt4dftu.common.configuration import Config
from BayesOpt4dftu.common.logger import BoLoggerGenerator
from BayesOpt4dftu.core.delta_all import DeltaAll
from BayesOpt4dftu.utils.context_utils import task_timer


def parse_args():
    parser = argparse.ArgumentParser(
        description=f'Objective Function Calculator CLI. Use "./input.json" as the config file.')
    parser.add_argument('--version', action='version', version=f"{__package_name__} {__version__}")
    return parser.parse_args()


def main():
    _ = parse_args()

    driver_logger = BoLoggerGenerator.get_logger("Driver")
    driver_logger.info(f"{__package_name__}, Version: {__version__}")
    with task_timer("Task", driver_logger):
        # Initialize and read all configurations
        config = Config("input.json")
        DeltaAll.init_config(config)

        delta = DeltaAll()
        delta.compute_delta()
        delta.write_delta(to_stdout=True)

        delta.report_baseline_gap()
        if config.include_mag or config.print_magmom:
            delta.report_baseline_magnetization()


if __name__ == "__main__":
    main()

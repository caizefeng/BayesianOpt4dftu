import argparse

from BayesOpt4dftu.bo import *
from BayesOpt4dftu.configuration import Config
from BayesOpt4dftu.delta_all import DeltaAll
from BayesOpt4dftu.dft import VaspInit, DftManager
from BayesOpt4dftu.io_utils import TempFileManager, task_timer
from BayesOpt4dftu.logging import BoLoggerGenerator
from . import __version__


def main():
    parser = argparse.ArgumentParser(description="BayesOpt4dftu CLI tool")
    parser.add_argument('--version', action='version', version=f"BayesOpt4dftu {__version__}")
    args = parser.parse_args()  # This will print version and exit if --version is provided

    driver_logger = BoLoggerGenerator.get_logger("Driver")
    with task_timer("Task", driver_logger):

        # Initialize and read all configurations
        config = Config("input.json")
        TempFileManager.init_config(config)
        DftManager.init_config(config)
        VaspInit.init_config(config)
        DeltaAll.init_config(config)
        BoDftuIterator.init_config(config)

        dft_manager = DftManager()
        if config.dry_run:
            driver_logger.info("Dry run set to True.")
            if not config.dftu_only:
                dft_manager.run_task(method='hse')
            dft_manager.run_task(method='dftu')
        else:
            driver_logger.info("Dry run set to False.")

            temp_file_manager = TempFileManager()
            temp_file_manager.setup_temp_files()

            if not config.dftu_only:
                dft_manager.run_task(method='hse')

            bo_iterator = BoDftuIterator()
            delta = DeltaAll()
            for i in range(config.iteration):
                dft_manager.run_task(method='dftu')
                delta.compute_delta()
                delta.write_delta()

                # Print baseline band gap for reference
                if i == 0:
                    if config.baseline == 'hse':
                        driver_logger.info(f"Band gap from hybrid DFT calculation: {delta.dg.get_baseline_gap()} eV")
                    elif config.baseline == 'gw':
                        driver_logger.info(f"Band gap from GW calculation: {delta.dg.get_baseline_gap()} eV")

                bo_iterator.next()
                if bo_iterator.converge():
                    break

            # Visualize the optimization process and retrieve optimized U
            bo_iterator.plot()
            optimal_u, _ = bo_iterator.get_optimal()

            if config.get_optimal_band:
                bo_iterator.update_u_config(optimal_u)
                dft_manager.run_task(method='dftu')
                delta.compute_delta()
                delta.write_delta()
                driver_logger.info("An additional DFT+U calculation using optimal U values performed.")

            dft_manager.finalize()
            bo_iterator.finalize()
            temp_file_manager.clean_up()


if __name__ == "__main__":
    main()

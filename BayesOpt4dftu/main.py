import argparse

from BayesOpt4dftu.bo import *
from BayesOpt4dftu.configuration import Config
from BayesOpt4dftu.delta_all import DeltaAll
from BayesOpt4dftu.dft import VaspInit, DftExecutor
from BayesOpt4dftu.io_utils import TempFileManager, BoLoggerGenerator
from . import __version__


def main():
    parser = argparse.ArgumentParser(description="BayesOpt4dftu CLI tool")
    parser.add_argument('--version', action='version', version=f"BayesOpt4dftu {__version__}")
    args = parser.parse_args()  # This will print version and exit if --version is provided

    logger_generator = BoLoggerGenerator()
    driver_logger = logger_generator.get_logger("Driver")
    dft_logger = logger_generator.get_logger("DFT")
    bo_logger = logger_generator.get_logger("Bayesian Optimization")

    driver_logger.info("Task begins.")
    driver_logger.info(f"Loading configuration ...")

    # Initialize and read all configurations
    config = Config("input.json")
    dft_executor = DftExecutor(config)
    VaspInit.init_config(config)
    DeltaAll.init_config(config)
    BayesOptDftu.init_config(config)

    driver_logger.info(f"Configuration loaded from file {config.config_file_name}.")
    dft_logger.info("DFT calculations begin.")

    if config.dry_run:
        dft_logger.info("Dry run set to True.")
        if not config.dftu_only:
            dft_executor.calculate(method='hse')
        dft_executor.calculate(method='dftu')
        dft_logger.info("No actual calculations were performed. Review the input files before proceeding.")
        dft_logger.info("Dry run executed.")
    else:
        dft_logger.info("Dry run set to False.")

        temp_file_manager = TempFileManager(config)
        temp_file_manager.setup_temp_files()
        driver_logger.info("Temporary files initiated.")

        if not config.dftu_only:
            dft_logger.info("Hybrid DFT calculation begins.")
            dft_executor.calculate(method='hse')
            dft_logger.info("Hybrid DFT calculation finished.")

        dft_logger.info("DFT+U calculations begin.")
        bo_logger.info("Bayesian Optimization begins.")

        obj = 0
        bayesian_opt = BayesOptDftu()
        for i in range(config.iteration):
            dft_executor.calculate(method='dftu')

            delta = DeltaAll()
            delta.compute_delta()
            delta.write_delta()

            # Print baseline band gap for reference
            if i == 0:
                if config.baseline == 'hse':
                    dft_logger.info(f"Band gap from hybrid DFT calculation: {delta.dg.get_baseline_gap()} eV")
                elif config.baseline == 'gw':
                    dft_logger.info(f"Band gap from GW calculation: {delta.dg.get_baseline_gap()} eV")

            obj_next = bayesian_opt.next()
            if BayesOptDftu.converge(obj_next, obj):
                bo_logger.info(f"Convergence reached at iteration {i + 1}, exiting.")
                break
            obj = obj_next

        # Visualize the optimization process and retrieve optimized U
        bayesian_opt.plot()
        optimal_u, optimal_obj = bayesian_opt.get_optimal()
        bo_logger.info(f"Optimal U value: {optimal_u}")
        bo_logger.info(f"Optimal objective function: {optimal_obj}")
        bo_logger.info("Bayesian Optimization finished.")

        if config.get_optimal_band:
            bayesian_opt.update_u_config(optimal_u)
            dft_executor.calculate(method='dftu')

            delta = DeltaAll()
            delta.compute_delta()
            delta.write_delta()
            dft_logger.info("An additional DFT+U calculation using optimal U values performed.")

        dft_logger.info("DFT+U calculations finished.")
        dft_logger.info("All DFT calculations finished.")

        temp_file_manager.clean_up_temp_files()
        driver_logger.info("Temporary files removed.")

    driver_logger.info("Task completed.")


if __name__ == "__main__":
    main()

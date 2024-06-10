from BayesOpt4dftu import __package_name__, __version__
from BayesOpt4dftu.cli.main_cli import parse_args
from BayesOpt4dftu.common.configuration import Config
from BayesOpt4dftu.common.logger import BoLoggerGenerator
from BayesOpt4dftu.core.bo import BoDftuIterator
from BayesOpt4dftu.core.delta_all import DeltaAll
from BayesOpt4dftu.core.dft import VaspInit, DftManager
from BayesOpt4dftu.core.temp_file_manager import TempFileManager
from BayesOpt4dftu.utils.context_utils import task_timer


def main():
    _ = parse_args()  # This will print version and exit if --version is provided

    driver_logger = BoLoggerGenerator.get_logger("Driver")
    driver_logger.info(f"{__package_name__}, Version: {__version__}")
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

            if config.resume_checkpoint:
                driver_logger.info("Optimization resumed from saved temporary files as the checkpoint.")
                driver_logger.info("Please ensure the settings are consistent.")

            temp_file_manager = TempFileManager()
            temp_file_manager.setup_temp_files()

            if not config.dftu_only:
                dft_manager.run_task(method='hse')

            bo_iterator = BoDftuIterator()
            delta = DeltaAll()
            for i in range(config.iteration):  # the main BO loop
                dft_manager.run_task(method='dftu')
                delta.compute_delta()
                delta.write_delta()

                # Print baseline band gap for reference
                if i == 0:
                    delta.report_baseline_gap()
                    if config.include_mag:
                        delta.report_baseline_magnetization()

                bo_iterator.next()
                if bo_iterator.converge():
                    break

            # Visualize the optimization process and retrieve optimized U
            bo_iterator.plot()
            optimal_u, optimal_obj = bo_iterator.get_optimal()
            driver_logger.info(f"Optimal Hubbard U: {optimal_u}")
            driver_logger.info(f"Optimal objective function: {optimal_obj}")

            if config.get_optimal_band:
                bo_iterator.update_u_config(optimal_u)
                dft_manager.run_task(method='dftu')
                delta.compute_delta()
                delta.write_delta(na_padding=True)
                driver_logger.info("An additional DFT+U calculation using optimal U values performed and logged at "
                                   "the end.")
                delta.report_optimal_dftu_gap()
                if config.include_mag:
                    delta.report_optimal_dftu_magnetization()

            dft_manager.finalize()
            bo_iterator.finalize()
            temp_file_manager.clean_up()


if __name__ == "__main__":
    main()

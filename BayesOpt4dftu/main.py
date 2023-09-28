from BayesOpt4dftu.core import *
from BayesOpt4dftu.BoLogging import BoLogging
from BayesOpt4dftu.Config import Config


def main():
    logging_generator = BoLogging()
    driver_logger = logging_generator.get_logger("Driver")
    dft_logger = logging_generator.get_logger("DFT")
    bo_logger = logging_generator.get_logger("BayesianOptimization")

    driver_logger.info("Task begins.")
    driver_logger.info("Loading configuration from file input.json ...")

    # Initialize and read all configurations
    config = Config("input.json")

    driver_logger.info("Configuration loaded.")
    driver_logger.info("DFT calculations begin.")

    if config.dry_run:
        dft_logger.info("Dry run set to True.")
        if not config.dftu_only:
            calculate(command=config.vasp_run_command, outfilename=config.out_file_name, method='hse',
                      import_kpath=config.import_kpath,
                      is_dry=True)
        calculate(command=config.vasp_run_command, outfilename=config.out_file_name, method='dftu',
                  import_kpath=config.import_kpath,
                  is_dry=True)
        dft_logger.info("No actual calculations were performed. Review the input files before proceeding.")
        driver_logger.info("Dry run executed.")
    else:
        dft_logger.info("Dry run set to False.")

        header = []
        for i, u in enumerate(config.which_u):
            header.append('U_ele_%s' % str(i + 1))

        if os.path.exists('./u_tmp.txt'):
            os.remove('./u_tmp.txt')

        with open('./u_tmp.txt', 'w+') as f:
            f.write('%s band_gap delta_band \n' % (' '.join(header)))

        dft_logger.info("Production run initiated.")

        if not config.dftu_only:
            dft_logger.info("Hybrid DFT calculation begins.")
            calculate(command=config.vasp_run_command, outfilename=config.out_file_name, method='hse',
                      import_kpath=config.import_kpath,
                      is_dry=False)
            dft_logger.info("Hybrid DFT calculation finished.")

        driver_logger.info("Bayesian Optimization begins.")
        dft_logger.info("GGA+U calculations begin.")

        obj = 0
        for i in range(config.iteration):
            calculate(command=config.vasp_run_command, outfilename=config.out_file_name, method='dftu',
                      import_kpath=config.import_kpath,
                      is_dry=False)
            db = delta_band(bandrange=config.br, path='./')
            db.deltaBand()

            bayesian_opt = bayesOpt_DFTU(path='./', opt_u_index=config.which_u, u_range=config.urange, kappa=config.k,
                                         a1=config.a1, a2=config.a2,
                                         elements=config.elements)
            obj_next = bayesian_opt.bo()
            if abs(obj_next - obj) <= config.threshold:
                bo_logger.info("Convergence reached, exiting.")
                break
            obj = obj_next

        dft_logger.info("GGA+U calculations finished.")
        driver_logger.info("DFT calculations finished.")
        bayesian_opt.plot()

        tmp_tuple = bayesian_opt.optimal
        bo_logger.info(f"Optimal U value: {tmp_tuple[0]}")
        bo_logger.info(f"Optimal objective function: {tmp_tuple[1]}")

        driver_logger.info("Bayesian Optimization finished.")

        os.system('mv ./u_tmp.txt ./u_kappa_%s_a1_%s_a2_%s.txt' % (config.k, config.a1, config.a2))

    driver_logger.info("Task completed.")


if __name__ == "__main__":
    main()

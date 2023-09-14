from BayesOpt4dftu.core import *


def main():
    with open("input.json", "r") as f:
        data = json.load(f)

    vasp_env_params = data['vasp_env']
    # Command to run VASP executable.
    vasp_run_command = vasp_env_params.get('vasp_run_command', 'srun -n 54 vasp_ncl')
    # Define the name for output file.
    out_file_name = vasp_env_params.get('out_file_name', 'vasp.out')
    # Define the path direct to the VASP pseudopotential.
    vasp_pp_path = vasp_env_params.get('vasp_pp_path', '/home/maituoy/pp_vasp/')
    dry_run = vasp_env_params.get('dry_run', False)

    bo_params = data['bo']
    k = float(bo_params.get('kappa', 5))
    a1 = bo_params.get('alpha1', 0.25)
    a2 = bo_params.get('alpha2', 0.75)
    which_u = tuple(bo_params.get('which_u', [1, 1]))
    urange = tuple(bo_params.get('urange', [-10, 10]))
    br = tuple(bo_params.get('br', [5, 5]))
    import_kpath = bo_params.get('import_kpath', False)
    elements = bo_params.get('elements', ['In', 'As'])
    iteration = bo_params.get('iteration', 50)
    threshold = bo_params.get('threshold', 0.0001)

    os.environ['VASP_PP_PATH'] = vasp_pp_path

    header = []
    for i, u in enumerate(which_u):
        header.append('U_ele_%s' % str(i + 1))

    if os.path.exists('./u_tmp.txt'):
        os.remove('./u_tmp.txt')

    with open('./u_tmp.txt', 'w+') as f:
        f.write('%s band_gap delta_band \n' % (' '.join(header)))

    if dry_run:
        calculate(command=vasp_run_command, outfilename=out_file_name, method='hse', import_kpath=import_kpath,
                  is_dry=True)
        calculate(command=vasp_run_command, outfilename=out_file_name, method='dftu', import_kpath=import_kpath,
                  is_dry=True)
        print("Dry run executed. No actual calculations were performed. ")
        print("Review the input files before proceeding.")

    else:
        calculate(command=vasp_run_command, outfilename=out_file_name, method='hse', import_kpath=import_kpath,
                  is_dry=False)
        obj = 0
        for i in range(iteration):
            calculate(command=vasp_run_command, outfilename=out_file_name, method='dftu', import_kpath=import_kpath,
                      is_dry=False)
            db = delta_band(bandrange=br, path='./')
            db.deltaBand()

            bayesian_opt = bayesOpt_DFTU(path='./', opt_u_index=which_u, u_range=urange, kappa=k, a1=a1, a2=a2,
                                         elements=elements)
            obj_next = bayesian_opt.bo()
            if abs(obj_next - obj) <= threshold:
                print("Optimization has been finished!")
                break
            obj = obj_next

        bayesian_opt.plot()
        print(bayesian_opt.optimal)

        os.system('mv ./u_tmp.txt ./u_kappa_%s_a1_%s_a2_%s.txt' % (k, a1, a2))


if __name__ == "__main__":
    main()

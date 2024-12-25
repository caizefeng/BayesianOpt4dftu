# BayesianOpt4dftu #

![version](https://img.shields.io/badge/version-2.6.6-blue)

Determine the Hubbard U parameters in DFT+U using the Bayesian Optimization approach.

## Prerequisites

### Python Version

- Tested on Python 3.8

### Python Dependencies
These will be installed automatically when you install the package via pip:

- [`numpy`](https://numpy.org)
- [`pandas`](https://pandas.pydata.org)
- [`ase`](https://wiki.fysik.dtu.dk/ase)
- [`pymatgen`](https://pymatgen.org)
- [`bayesian-optimization`](https://github.com/fmfn/BayesianOptimization)
- [`vaspvis`](https://github.com/DerekDardzinski/vaspvis)

### Other Dependencies
These dependencies need to be installed manually:

- [Vienna Ab initio Simulation Package (VASP)](https://www.vasp.at)

## Installation

To install `BayesianOpt4dftu`, use the following command:

```shell
pip install git+https://github.com/caizefeng/BayesianOpt4dftu.git
```

## Usage

For the demonstration, 
let's focus on fitting the Hubbard U values for both elements in indium arsenide (InAs). 

The example input and output can be found in the `/examples/2d` directory:

### 1. Edit `input.json`

Navigate to the example directory:

```shell
cd examples/2d
``` 

Adjust the `input.json` file with the appropriate `vasp_env` settings based on your system specifications and the location of your VASP binary.

### 2. Execute

Run the following command:

```shell
bo_dftu
```

### 3. Results

Upon reaching the threshold or maximum iterations, two output files are generated:

- `u_xxx.txt`/`formatted_u_xxx.txt`: Contains U parameters, band gap, Δgap, Δband, and Δmagnetization (optional) for each iteration.
- `1D_xxx.png` or `2D_xxx.png`: Provides a visual representation of the Gaussian process predicted mean and acquisition function. 
   This file will be omitted if you set three or more optimizable U parameters.

**Example of BO plots**:

- 1-D Bayesian Optimization for Ge

  <img src="https://github.com/caizefeng/BayesianOpt4dftu/blob/master/examples/1d/1D_kappa_5.0_ag_0.5_ab_0.5_am_0.0.png" width="600" height="450">

- 2-D Bayesian Optimization for InAs

  <img src="https://github.com/caizefeng/BayesianOpt4dftu/blob/master/examples/2d/2D_kappa_5.0_ag_0.25_ab_0.75_am_0.0.png" width="800" height="270">

Optimal U values are automatically deduced from the predicted mean space interpolation. 
Alternatively, you can use the `u_xxx.txt` file to select U values with the highest objective value.


## Configuration

Before running the program, configure the `input.json` file. It contains:

- **`vasp_env`**: Environment settings for VASP.

    - **`vasp_run_command`**:
        - **Description**: Running command for VASP executable.
        - **Example**: `"vasp_run_command": "mpirun -np 64 /path/to/vasp/executable"`

    - **`out_file_name`**:
        - **Description**: The desired name of the VASP output file.
        - **Example**: `"out_file_name": "slurm-vasp.out"`

    - **`vasp_pp_path`**:
        - **Description**: Path directing to the VASP pseudopotential. It should be the directory containing
          the `potpaw_PBE` folder.
        - **Example**: `"vasp_pp_path": "/path/to/pseudopotentials/"`

    - **`dry_run`**:
        - **Description**: Specifies if the run is a dry run (generating files only without actual computation) or not.
        - **Default**: `"dry_run": false`

    - **`dftu_only`**:
        - **Description**: Indicates whether only DFT+U is performed. If set to true, completed baseline calculations should be placed in the `<working dir>/<baseline>` directory.
        - **Default**: `"dftu_only": false`
          
    - **`get_optimal_band`**:
        - **Description**: Indicate if an additional DFT+U using optimal U values is performed after Bayesian optimization. 
          The results of this calculation will be appended to the end of the log file.
        - **Default**: `"get_optimal_band": true`


- **`bo`**: Settings specific to Bayesian Optimization.
    - **`resume_checkpoint`**:
        - **Description**: Determines whether the Bayesian Optimization resumes from a saved checkpoint, as defined by `u_tmp.txt` and `input_tmp.json`.
        - **Default**: `"resume_checkpoint": false`

    - **`baseline`**:
        - **Description**: Specifies the baseline calculation for Bayesian Optimization. Currently, only `"hse"` and `"gw"` are supported. `"gw"` must be executed separately, meaning it is only supported when `"dftu_only": true`.
        - **Default**: `"baseline": "hse"`

    - **`which_u`**:
        - **Description**: Specifies which element you'd like to optimize the U for.
        - Format: For a unary substance, it has to be `[1,]`. For
          compounds with over 2 elements, you can set each element to `0` or `1` to switch off/on the optimization for that
          element.
        - **Example**: For InAs, when optimizing for both In and As, it should be set as `"which_u": [1, 1]`

    - **`br`**:
        - **Description**: Specifies the band range you'd like to include in your Δband.
        - Format: A list of two integers, defining the number of valence bands and conduction bands from the Fermi
          level.
        - **Default**: `"br": [5, 5]`

    - **`kappa`**:
        - **Description**: Controls the balance between exploration and exploitation when the acquisition function samples the next points.
          A lower value (nearing `0`) indicates a preference for exploitation. A higher value (approaching `10`) indicates a preference for exploration.
        - **Default**: `"kappa": 5`

    - **`alpha_gap`** and **`alpha_band`**:
        - **Description**: Specifies the weight coefficients of Δgap and Δband respectively. So far, there are no constraints on the choice of `alpha`s, meaning they can be arbitrary positive real numbers.
        - **Default**: `"alpha_gap": 0.25` and `"alpha_band": 0.75`

    - **`alpha_mag`**:
        - **Description**: Specifies the weight coefficients of Δmagnetization. `LORBIT` must be set in all `INCAR` files. A `alpha_mag` of `0` will exclude Δmagnetization from the loss function.
        - **Default**: `"alpha_mag": 0.0`

    - **`mag_axis`**:
        - **Description**: Specifies the Cartesian component of the non-collinear magnetic moment used to calculate Δmagnetization. Available options are `"x"`, `"y"`, `"z"`, and `"all"`(all 3 components). This parameter is only applicable to non-collinear calculations and has no effect on collinear calculations (when `LNONCOLLINEAR=.FALSE.` and `LSORBIT=.FALSE.`).
        - **Default**: `"mag_axis": "all"`

    - **`threshold`**:
        - **Description**: Specifies the desired accuracy for the objective function, measured as the difference between values from two consecutive iterations, at which you'd like to stop the BO process. 
          A `threshold` of `0.0` (and `threshold_opt_u` of `0.0` as well) will disable the convergence assessment, meaning the BO will exit only upon reaching the maximum iterations.
        - **Default**: `"threshold": 0.0001`

    - **`urange`**:
        - **Description**: Specifies the U parameter range for optimization. The unit is eV. Defining different U ranges for separate
          elements is currently unsupported. 
        - **Default**: `"urange": [-10, 10]`

    - **`elements`**:
        - **Description**: Lists the elements in your system. This is used for plotting the BO results. If it's a
          unary substance, it has to be `["ele",]`.
        - **Example**: `"elements": ["In", "As"]`

    - **`iteration`**:
        - **Description**: Sets the maximum iterations that BO will perform.
        - **Default**: `"iteration": 50`

    - **`report_optimum_interval`**:
        - **Description**: Sets the interval (in iterations) at which the optimal Hubbard U values are calculated and logged.
        - **Default**: `"report_optimum_interval": 10`
     
    - **`threshold_opt_u`**:
        - **Description**: This criterion supplements the original `threshold` parameter and can be used as the sole convergence criterion or in conjunction with `threshold`. Specifies the desired accuracy for the optimal Hubbard U values, measured as the difference between values from two iterations separated by `report_optimum_interval`. 
        A `threshold_opt_u` of `0.0` will disable this optimal U-based convergence assessment.
        - **Default**: `"threshold_opt_u": 0.0`

    - **`print_magmom`**:
        - **Description**: Specifies whether to print the magnetic moment at each iteration.
        - **Default**: `"print_magmom": false`

- **`structure_info`** : Includes geometry information (such as lattice parameter, lattice vectors, atomic position,
  etc.) of the target materials.
  #### An example of InAs:
    - **`lattice_param`** and **`cell`**: Specify the 2nd to 5th rows in your POSCAR.
        ```json
        {
            "lattice_param": 6.0584,
            "cell": [
                [
                    0.0,
                    0.5,
                    0.5
                ],
                [
                    0.5,
                    0.0,
                    0.5
                ],
                [
                    0.5,
                    0.5,
                    0.0
                ]
            ]
        }
        ```

    - **`atoms`**: Specifies the atomic positions of each atom in your system and the initial magnetic moment if there is
      any.

      Non-collinear magnetism:
        ```json
        {
            "atoms": [
                [
                    "In",
                    [
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        1e-06
                    ]
                ],
                [
                    "As",
                    [
                        0.75,
                        0.75,
                        0.75
                    ],
                    [
                        0,
                        0,
                        1e-06
                    ]
                ]
            ]
        }
        ```

      Collinear magnetism:
        ```json
        {
            "atoms": [
                [
                    "In",
                    [
                        0,
                        0,
                        0
                    ],
                    1e-06
                ],
                [
                    "As",
                    [
                        0.75,
                        0.75,
                        0.75
                    ],
                    1e-06
                ]
            ]
        }
        ```

      In this case, there are two atoms in the primitive cell at positions `[0, 0, 0]`
      and `[0.75, 0.75, 0.75]`. 
      The second term under each atom specifies the initial magnetic moment. 
      For calculations excluding non-collinear magnetization or spin-orbit coupling, this is an integer (or it could be a one-dimensional array with one element).
      Otherwise, it's a one-dimensional array with three elements, each representing the initial moment in a specific Cartesian direction.
      To avoid omission errors in the `ASE` package, the initial moment should be set to a small, non-zero number if it is intended to be 0.
    - **`kgrid_hse`** and **`kgrid_pbe`**:
        - **Description**: Set the self-consistent k-point grid for HSE and PBE+U calculations, respectively.
        - **Example**: `"kgrid_pbe": [7, 7, 7]` specifies a 7x7x7 k-point grid for PBE+U calculation.
    - **`num_kpts`** and **`kpath`**:
        - **Description**: Specify the non-self-consistent (non-SC) k-point path for band structure calculations.
          `num_kpts` can be either an integer or a string "auto".
        - **Example**: `"num_kpts": 50` and `"kpath": "G X W L G K"` set the k-point path to `G-X-W-L-G-K` and the number of k-points per path segment to 50. Important: In this mode, contributions to Δband are weighted to achieve an approximately uniform density of sampling along the path.
         `"num_kpts": "auto"` automatically determines the path and number of k-points based on the HSE or GW baseline calculation.
    - **`custom_kpoints`**:
        - **Description**: Specifies the custom reciprocal coordinates for certain k-points in `kpath`. If it conflicts with common BZ coordinate conventions for certain k-points, this setting takes higher priority.
        - **Example**: `"custom_kpoints": null` (the default) or `"custom_kpoints": {"H": [0.5, -0.5, 0.5],  "F": [0.5, 0.5, 0]}`
    - **`custom_POTCAR_path`**:
        - **Description**: Specifies the path to the custom POTCAR file. This file will be copied to all VASP working directories.
        - **Example**: `"custom_POTCAR_path": null` (the default) or `"custom_POTCAR_path": "./POTCAR_Pb_sv_Te_sv"`

- **`general_flags`**: General INCAR tags required in all VASP calculations.
- **`scf`**: INCAR tags that will only be added in SCF calculations.
- **`band`**: INCAR tags that will only be added in band structure calculations.
- **`pbe`**: INCAR tags that will only be added when using PBE as exchange-correlation functional.
- **`hse`**: INCAR tags that will only be added when using HSE06 as exchange-correlation functional.
  
  Check the [ASE VASP calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html) documentation for additional tag keys.
  (A large portion of them are simply the lowercase versions of the corresponding INCAR tags.)

## Citation

Please cite the following work if you use this code.

[1] M. Yu, S. Yang, C. Wu, N. Marom, Machine learning the Hubbard U parameter in DFT+ U using Bayesian optimization, npj
Computational Materials, 6(1):1–6, 2020.


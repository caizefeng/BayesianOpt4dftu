# BayesianOpt4dftu #

![version](https://img.shields.io/badge/version-1.0.1-blue)

Determine the Hubbard U parameters in DFT+U using the Bayesian Optimization approach.

## Prerequisites

1. Python 3.6 or higher
2. NumPy
3. Pandas
4. ASE (https://wiki.fysik.dtu.dk/ase)
5. pymatgen (https://pymatgen.org)
6. bayesian-optimization (https://github.com/fmfn/BayesianOptimization)
7. Vienna Ab initio Simulation Package (VASP) (https://www.vasp.at)
8. Vaspvis (https://github.com/DerekDardzinski/vaspvis)

## Configuration

Before running the program, configure the `input.json` file. It contains:

- **`vasp_env`**: Environment settings for VASP.

    - **`vasp_run_command`**:
        - Description: Running command for VASP executable.
        - Example: `"vasp_run_command": "mpirun -np 64 /path/to/vasp/executable"`

    - **`out_file_name`**:
        - Description: The desired name of the VASP output file.
        - Example: `"out_file_name": "slurm-vasp.out"`

    - **`vasp_pp_path`**:
        - Description: Path directing to the VASP pseudopotential. It should be the directory containing
          the `potpaw_PBE` folder.
        - Example: `"vasp_pp_path": "/path/to/pseudopotentials/"`

    - **`dry_run`**:
        - Description: Specifies if the run is a dry run (generating files only without actual computation) or not.
        - Example: `"dry_run": false`

    - **`dftu_only`**:
        - Description: Indicates whether only DFT+U is performed. If set to true, completed baseline calculations should be placed in the `<working dir>/<baseline>` directory.
        - Example: `"dftu_only": false`
          
    - **`get_optimal_band`**:
        - Description: Indicate if an additional DFT+U using optimal U values is performed after Bayesian optimization. 
          The results of this calculation will be appended to the end of the log file.
        - Example: `"get_optimal_band": false`


- **`bo`**: Settings specific to Bayesian Optimization.

    - **`baseline`**:
        - Description: Specifies the baseline calculation for Bayesian Optimization. Currently, only `"hse"` and `"gw"` are supported. `"gw"` must be executed separately, meaning it is only supported when `"dftu_only": true`.
        - Example: `"baseline": "hse"`

    - **`which_u`**:
        - Description: Specifies which element you'd like to optimize the U for.
        - Format: For a unary substance, it has to be `[1,]`. For
          compounds with over 2 elements, you can set each element to `0` or `1` to switch off/on the optimization for that
          element.
        - Example: For InAs, when optimizing for both In and As, it should be set as `"which_u": [1, 1]`

    - **`br`**:
        - Description: Specifies the band range you'd like to include in your Δband.
        - Format: A list of two integers, defining the number of valence bands and conduction bands from the Fermi
          level.
        - Example: `"br": [5, 5]`

    - **`kappa`**:
        - Description: Controls the balance between exploration and exploitation when the acquisition function samples the next points.
          A lower value (nearing `0`) indicates a preference for exploitation. A higher value (approaching `10`) indicates a preference for exploration.
        - Example: `"kappa": 5`

    - **`alpha_gap`** and **`alpha_band`**:
        - Description: Specifies the weight coefficients of Δgap and Δband respectively. So far, there are no constraints on the choice of `alpha`s, meaning they can be arbitrary positive real numbers.
        - Examples: `"alpha_gap": 0.25` and `"alpha_band": 0.75`

    - **`alpha_mag`**:
        - Description: Specifies the weight coefficients of Δmagnetization. `LORBIT` must be set in all `INCAR` files. A `alpha_mag` of `0` will exclude Δmagnetization from the loss function.
        - Example: `"alpha_mag": 0.1`

    - **`threshold`**:
        - Description: Specifies the accuracy at which you'd like to stop the BO process. 
          A `threshold` of `0` will disable convergence assessment, meaning the BO will exit only upon reaching the maximum steps.
        - Example: `"threshold": 0.0001`

    - **`urange`**:
        - Description: Specifies the U parameter range for optimization. The unit is eV. Defining different U ranges for separate
          elements is currently unsupported. 
        - Example: `"urange": [-10, 10]`

    - **`elements`**:
        - Description: Lists the elements in your system. This is used for plotting the BO results. If it's a
          unary substance, it has to be `["ele",]`.
        - Example: `"elements": ["In", "As"]`

    - **`iteration`**:
        - Description: Sets the maximum steps that BO will perform.
        - Example: `"iteration": 50`


- **`structure_info`** : Includes geometry information (such as lattice parameter, lattice vectors, atomic position,
  etc.) of the target materials.
  #### Example based on InAs:
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

    - **`atoms`**: Specify the atomic positions of each atom in your system and the initial magnetic moment if there is
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

      So in this case, there are two atoms in the primitive cell at positions `(0,0,0)`
      and `(0.75, 0.75, 0.75)`. The second term under each atom specifies the initial magnetic moment. 
      For calculations excluding non-collinear magnetization or spin-orbit coupling, this is an integer; 
      otherwise, it's a (3,) array, with each element representing the initial moment in a specific direction. 
      To avoid omission error in the ASE package, the initial moment should be set to a small, non-zero number if it is intended to be 0.
    - **`kgrid_hse`** and **`kgrid_pbe`**:
        - Description: Set the self-consistent k-point grid for HSE and PBE+U calculations, respectively.
        - Example: `"kgrid_pbe": [7, 7, 7]` specifies a 7x7x7 k-point grid for PBE+U calculation.
    - **`num_kpts`** and **`kpath`**:
        - Description: Specify the non-self-consistent (non-SC) k-point path for band structure calculations.
          `num_kpts` can be either an integer or a string "auto".
        - Example: `"num_kpts": 50` and `"kpath": "G X W L G K"` set the k-point path to `G-X-W-L-G-K` and the number of k-points per path segment to 50. Important: In this mode, contributions to Δband are weighted to achieve an approximately uniform density of sampling along the path.
         `"num_kpts": "auto"` automatically determines the path and number of k-points based on the HSE or GW baseline calculation.
                    

- **`general_flags`**: General INCAR flags required in all VASP calculation.
- **`scf`**: Flags that will only be added in SCF calculation.
- **`band`**: Flags that will only be added in band structure calculation.
- **`pbe`**: Flags that will only be added when using PBE as exchange-correlation functional.
- **`hse`**: Flags that will only be added when using HSE06 as exchange-correlation functional.
  
    Check ASE VASP calculator documentation for additional flag keys.

## Installation

```shell
pip install git+https://github.com/caizefeng/BayesianOpt4dftu.git
```

## Usage

For demonstration, consider the `/example/2d`:

### 1. Edit `input.json`

Change to the example directory:

```shell
cd example/2d
```

Update the `input.json` file with the appropriate `vasp_env` settings based on your system specifications and the location of your VASP binary.

### 2. Execute

Run the following command:

```shell
bo_dftu
```

### 3. Results

Upon reaching the threshold or maximum iterations, two output files are generated:

- `u_xxx.txt`: Contains U parameters, band gap, Δgap, Δband, and Δmagnetizaion (optional) for each step.
- `1D_xxx.png` or `2D_xxx.png`: Provides a visual representation of the Gaussian process predicted mean and acquisition function. 
   This file will be omitted if you set three or more optimizable U parameters.

**Example of BO plots**:

- 1-D Bayesian Optimization for Ge

  <img src="https://github.com/caizefeng/BayesianOpt4dftu/blob/master/example/1d/1D_kappa_5.0_ag_0.5_ab_0.5_am_0.0.png" width="600" height="450">

- 2-D Bayesian Optimization for InAs

  <img src="https://github.com/caizefeng/BayesianOpt4dftu/blob/master/example/2d/2D_kappa_5.0_ag_0.25_ab_0.75_am_0.0.png" width="800" height="270">

Optimal U values are automatically deduced from the predicted mean space interpolation. 
Alternatively, you can use the `u_xxx.txt` file to select U values with the highest objective value.

## Citation

Please cite the following work if you use this code.

[1] M. Yu, S. Yang, C. Wu, N. Marom, Machine learning the Hubbard U parameter in DFT+ U using Bayesian optimization, npj
Computational Materials, 6(1):1–6, 2020.


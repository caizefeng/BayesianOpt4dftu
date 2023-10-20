# BayesianOpt4dftu #

This code determines the Hubbard U parameters in DFT+U via Bayesian Optimization approach.

## Requirements ##

1. Python 3.6+
2. NumPy
3. Pandas
4. ASE (https://wiki.fysik.dtu.dk/ase/)
5. pymatgen (https://pymatgen.org/)
6. bayesian-optimization https://github.com/fmfn/BayesianOptimization
7. Vienna Ab initio Simulation Package (VASP) https://www.vasp.at/
8. Vaspvis (https://github.com/DerekDardzinski/vaspvis)

## Set up the input file (`input.json`) before running the code

The input file contains these parts:

- **`vasp_env`**: Environment settings for VASP.

    - **`vasp_run_command`**:
        - Description: Running command for VASP executable.
        - Example: `"vasp_run_command": "mpirun -np 64 /path/to/vasp/executable"`

    - **`out_file_name`**:
        - Description: The name of the VASP output file.
        - Example: `"out_file_name": "slurm-vasp.out"`

    - **`vasp_pp_path`**:
        - Description: Path directing to the VASP pseudopotential. It should be the directory containing
          the `potpaw_PBE` folder.
        - Example: `"vasp_pp_path": "/path/to/pseudopotentials/"`

    - **`dry_run`**:
        - Description: Specifies if the run is a dry run (generate files only without actual computation) or not.
        - Example: `"dry_run": false`

    - **`dftu_only`**:
        - Description: Indicate if only DFT+U is performed.
        - Example: `"dftu_only": false"`
          
    - **`get_optimal_band`**:
        - Description: Indicate if an additional DFT+U using optimal U values is performed after Bayesian optimization.
        - Example: `"get_optimal_band": false"`


- **`bo`**: Settings specific to Bayesian Optimization.

    - **`which_u`**:
        - Description: Defines which element you'd like to optimize the U for.
        - Format: For a unary substance, it has to be `(1,)`. For
          compounds with over 2 elements, you can set each element to 0 or 1 to switch off/on the optimization for that
          element.
        - Example: For InAs, when optimizing for both In and As, it should be set as `"which_u": [1, 1]`.

    - **`br`**:
        - Description: Specifies the band range you'd like to include in your Δband.
        - Format: A list of two integers, defining the number of valence bands and conduction bands from the Fermi
          level.
        - Example: `"br": [5, 5]`.

    - **`kappa`**:
        - Description: Controls the exploration and exploitation when the acquisition function samples the next points.
          Exploitation 0 <--kappa --> 10 Exploration
        - Example: `"kappa": 5`.

    - **`alpha1`** and **`alpha2`**:
        - Description: Specifies the weight coefficients of Δgap and Δband respectively.
        - Examples: `"alpha1": 0.25` and `"alpha2": 0.75`.

    - **`threshold`**:
        - Description: Specifies the accuracy at which you'd like to stop the BO process. Note: A threshold of 0 will disable convergence assessment.
        - Example: `"threshold": 0.0001`.

    - **`urange`**:
        - Description: Defines the U parameter range for optimization. Note: Defining different U ranges for separate
          elements is unsupported.
        - Example: `"urange": [-10, 10]`.

    - **`import_kpath`**:
        - Description: Provides an external list of high-symmetry k-points if some special k coordinates aren't
          available in
          the ASE library.
        - Example: `"import_kpath": false`.

    - **`elements`**:
        - Description: Lists the elements in your system. This is used for plotting the BO results. Note: If it's a
          unary
          substance, it has to be [ele,].
        - Example: `"elements": ["In", "As"]`.

    - **`iteration`**:
        - Description: Sets the maximum steps that BO will perform.
        - Example: `"iteration": 50`.


- **`structure_info`** : Includes geometry information (such as lattice parameter, lattice vectors, atomic position,
  etc.) of the target materials.
  #### Example based on InAs:
    - **`lattice_param`** and **`cell`**: define the 2nd to 5th rows in your POSCAR.
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

    - **`atoms`**: Define the atomic positions of each atom in your system and the initial magnetic moment if there is
      any.

      With SOC:
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

      Without SOC:

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

      So in this case, there are two atoms in the primitive cell which are located at the position `(0,0,0)`
      and `(0.75, 0.75, 0.75)`. The second term under each atom defines the initial magnetic moment. If the spin-orbit
      coupling is not included in your calculation, it is just an integer while otherwise it is a (3,) array of each
      element defines the initial moment of corresponding direction. If the initial moment is 0, it has to be set to a
      small number to avoid conflict in the ASE.

- **`general_flags`**: Includes general flags required in the VASP calculation.
- **`scf`**: Flags required particularly in SCF calculation.
- **`band`**: Flags required particularly in band structure calculation.
- **`pbe`**: Flags required when using PBE as exchange-correlation functional.
- **`hse`**: Flags required when using HSE06 as exchange-correlation functional.
  The flags can be added or removed. More flag keys can be found in the ASE VASP calculator.

## Installation

```shell
pip install BayesOpt4dftu
```

## Usage

I will use `/example/2d` as an example:

#### 1. Modifying input.json

Navigate to the desired `example` directory

```shell
cd example/2d
```

Modify the `input.json` file with the appropriate environment settings `vasp_env` based on the specs of your system and
VASP binary.

#### 2. Running the code

After updating the respective sections, you can simply run the calculation by

```shell
bo_dftu
```

#### 4. Outputs

Once the threshold or the maximum iterations is reached, you will get two output files

`u_xx.txt` file consists of the U parameters, band gap, and the Δband at each step.

`1D_xxx.png` or `2D_xxx.png` plots showing you the Gaussian process predicted mean and the acquisition function.

Example of BO plots

  <img src="https://github.com/maituoy/BayesianOpt4dftu/blob/master/example/1d/1D_kappa_5_a1_0.5_a2_0.5.png" width="600" height="400">

  <img src="https://github.com/maituoy/BayesianOpt4dftu/blob/master/example/2d/2D_kappa_5_a1_0.25_a2_0.75.png" width="800" height="270">

Optimal U values will be output at the end of entire process based on the interpolation from the predicted mean space.
You can also pick up the Us that give you largest objective value from the u.txt file.

## Citation

Please cite the following work if you use this code.

[1] M. Yu, S. Yang, C. Wu, N. Marom, Machine learning the Hubbard U parameter in DFT+ U using Bayesian optimization, npj
Computational Materials, 6(1):1–6, 2020.


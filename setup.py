from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='BayesOpt4dftu',
    version='0.1.4',
    #    description='???',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Maituo Yu',
    #    author_email="???",
    url='https://github.com/caizefeng/BayesianOpt4dftu',
    packages=['BayesOpt4dftu'],
    install_requires=['numpy',
                      'matplotlib',
                      'ase==3.22.0',
                      'monty==2022.1.12.1',
                      'pyvista==0.37.0',
                      'pyprocar==5.6.6',
                      'pymatgen==2022.0.16',
                      'bayesian-optimization==1.4.2',
                      'pandas',
                      'vaspvis==1.2.2',
                      ],
    entry_points={
        'console_scripts': [
            'bo_dftu=BayesOpt4dftu.main:main',
        ],
    },
)

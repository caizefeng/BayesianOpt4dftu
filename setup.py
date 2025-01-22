import os
from typing import List

import setuptools


# Helper Functions

def read_file(rel_path: str) -> str:
    """Read the content of a file at a relative path."""
    here = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.join(here, rel_path)
    try:
        with open(abs_path, 'r', encoding='utf-8') as fp:
            return fp.read()
    except FileNotFoundError:
        raise RuntimeError(f"Unable to find the file at {abs_path}.")
    except IOError as e:
        raise RuntimeError(f"An error occurred while reading the file at {abs_path}: {e}")


def get_required_packages(file_path: str = 'requirements.txt') -> List[str]:
    """Retrieve the list of required packages from a requirements file."""
    return read_file(file_path).splitlines()


def get_version(rel_path: str) -> str:
    """Extract the version string from a Python file."""
    for line in read_file(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# Setup Configuration

setuptools.setup(
    name='BayesOpt4dftu',
    version=get_version("BayesOpt4dftu/__init__.py"),
    description='Bayesian Optimization toolkit for DFT+U',
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author='Maituo Yu',
    maintainer='Zefeng Cai',
    url='https://github.com/caizefeng/BayesianOpt4dftu',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    packages=setuptools.find_packages(),
    install_requires=get_required_packages(),
    python_requires='>=3.8',
    package_data={
        'BayesOpt4dftu': ['schemas/*.json']
    },
    entry_points={
        'console_scripts': [
            'bo_dftu=BayesOpt4dftu.main:main',
            'bo_formatter=BayesOpt4dftu.cli.formatter_cli:main',
            'bo_calculator=BayesOpt4dftu.cli.calculator_cli:main'
        ],
    },
)

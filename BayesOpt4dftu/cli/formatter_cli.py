import argparse

from BayesOpt4dftu import __package_name__
from BayesOpt4dftu.utils.file_utils import format_log_file, format_log_file_pd


def parse_args():
    parser = argparse.ArgumentParser(description=f'Log File Formatter for {__package_name__}.')
    parser.add_argument('input_file', type=str, help='Path to the input log file.')
    parser.add_argument('output_file', type=str, help='Path to the output log file.')
    parser.add_argument('-d', '--decimals', type=int, default=4, help='Number of decimal places to format the numbers.')
    parser.add_argument('-w', '--width', type=int, default=15, help='Width for each column.')
    return parser.parse_args()

def main():
    args = parse_args()
    format_log_file_pd(args.input_file, args.output_file, args.decimals, args.width)
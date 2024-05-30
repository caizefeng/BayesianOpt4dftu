import os
import shutil
import sys
import time
import warnings
from contextlib import contextmanager


@contextmanager
def task_timer(label, logger):
    logger.info(f"{label} begins.")
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed_time = end - start
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        hours_text = f"{int(hours)} Hour{'s' if hours != 1 else ''}"
        minutes_text = f"{int(minutes)} Minute{'s' if minutes != 1 else ''}"
        seconds_text = f"{seconds:.2f} Second{'s' if seconds != 1 else ''}"
        logger.info(f"{label} completed in {hours_text} {minutes_text} {seconds_text}.")


def modify_last_line_before_newline(file_path, additional_string):
    """
    Appends an additional string to the last line of a file before the newline character.

    Parameters:
    file_path (str): The path to the file to be modified.
    additional_string (str): The string to append to the last line.
    """
    with open(file_path, 'r+') as file:
        # Read all lines into a list
        lines = file.readlines()

        if lines:
            # Modify the last line
            lines[-1] = lines[-1].rstrip('\n') + " " + additional_string + "\n"

            # Go back to the start of the file
            file.seek(0)

            # Write all lines back to the file
            file.writelines(lines)


class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def find_and_readlines_first(directory, file_list, logger, extra_message=''):
    for filename in file_list:
        if os.path.exists(os.path.join(directory, filename)):
            with open(os.path.join(directory, filename), 'r') as file:
                return file.readlines()
    logger.error(f"None of these files ({file_list}) were found{' ' + extra_message if extra_message else ''}.")
    raise FileNotFoundError


def recreate_path_as_directory(path):
    # Remove the item at the path, whether it's a file, a directory, or something else
    if os.path.exists(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # Remove if it's a file or a link
        elif os.path.isdir(path):
            shutil.rmtree(path)  # Remove if it's a directory

    # Recreate the directory
    os.makedirs(path, exist_ok=True)


def error_handled_copy(source_path, target_path, logger, error_cause_message):
    try:
        # Check if the file is empty
        if os.path.getsize(source_path) == 0:
            # Log an error for the empty file
            logger.error(f"The file at {source_path}, required for subsequent calculations, is empty.")
            logger.error(f"This issue is likely because {error_cause_message}.")
            raise ValueError(f"Empty file error at {source_path}.")

        # Proceed with the copy if the file is not empty
        shutil.copy(source_path, target_path)

    except FileNotFoundError:
        logger.error(f"The file at {source_path}, required for subsequent calculations, is missing.")
        logger.error(f"This issue is likely because {error_cause_message}.")
        raise  # Re-raise the FileNotFoundError to halt the program
    except Exception as e:
        logger.error(f"An error occurred while copying from {source_path} to {target_path}. Error details: {e}.")
        raise  # Re-raise the caught exception to halt the program


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper  # noqa

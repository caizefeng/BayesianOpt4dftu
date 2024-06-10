import os
import sys
import time
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


class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

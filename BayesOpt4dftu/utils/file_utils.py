import os
import shutil

import pandas as pd


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


def format_log_file(input_file, output_file, decimals, width, logger=None):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    formatted_lines = []
    for line in lines:
        # Split line into components
        components = line.split()
        formatted_components = []

        for comp in components:
            try:
                # Try to convert to float and format, preserving scientific notation if needed
                num = float(comp)
                if 'e' in comp or 'E' in comp:
                    formatted_comp = f"{num:.{decimals}e}"
                else:
                    formatted_comp = f"{num:.{decimals}f}"
            except ValueError:
                # If conversion fails, just use the original component (e.g., for "N/A")
                formatted_comp = comp

            # Pad the formatted component to the specified width
            formatted_components.append(formatted_comp.ljust(width))

        # Join the formatted components back into a line
        formatted_line = ' '.join(formatted_components).strip()
        formatted_lines.append(formatted_line)

    with open(output_file, 'w') as outfile:
        outfile.write('\n'.join(formatted_lines))

    if logger:
        logger.info(f"Formatted log file saved to {output_file}")
    else:
        print(f"Formatted log file saved to {output_file}")


def format_log_file_pd(input_file, output_file, decimals, width, logger=None):
    # Load the data into a DataFrame
    with open(input_file, 'r') as infile:
        df = pd.read_csv(infile, delim_whitespace=True)

    # Function to format values to a fixed number of decimal points
    def format_value(val):
        try:
            num = float(val)
            return f"{num:.{decimals}e}" if 'e' in str(val) else f"{num:.{decimals}f}"
        except (ValueError, TypeError):
            return val

    # Apply formatting to all columns
    for col in df.columns:
        df[col] = df[col].apply(format_value)

    # Increase spacing between columns
    formatted_data = df.to_string(index=False, justify='right', col_space=width)

    # Write formatted data to the output file
    with open(output_file, 'w') as outfile:
        outfile.write(formatted_data)

    if logger:
        logger.info(f"Formatted log file saved to {output_file}")
    else:
        print(f"Formatted log file saved to {output_file}")

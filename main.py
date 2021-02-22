import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -- Configuration

# Endpoint input file: field separator
SEP = "\t"

# Endpoint input file: value encoding excluded controls
VAL_EXCL = "NA"

# At most how many lines are in the input file.
# User: the upper bound of 600_000 is ok, if you get an error
# message saying the bottom lines of the input file were not read,
# then increase this number by checking the result of
# "wc -l <input-file>".
# Dev: This is tricky. It's only needed to simplify the file parsing
# by allocating a large (i.e. at least the number of lines of the
# input file) array, that will be trimmed afterwards to keep only the
# lines of the input file.
MAX_LINES = 600_000

# Output files
FILE_CASES_CASES =       "n_cases_cases.parq"
FILE_CASES_CONTROLS =    "n_cases_controls.parq"
FILE_CASES_EXCL =        "n_cases_excl.parq"
FILE_CONTROLS_CASES =    "n_controls_cases.parq"
FILE_CONTROLS_CONTROLS = "n_controls_controls.parq"
FILE_CONTROLS_EXCL =     "n_controls_excl.parq"
FILE_EXCL_CASES =        "n_excl_cases.parq"
FILE_EXCL_CONTROLS =     "n_excl_controls.parq"
FILE_EXCL_EXCL =         "n_excl_excl.parq"
FILE_CORR_PHI =          "corr_phi.parq"
FILE_CORR_CHISQ =        "corr_chisq.parq"
FILE_CORR_CASE_RATIO =   "corr_case_ratio.parq"
FILE_CORR_OVERLAP_AB =   "corr_overlap_ab.parq"
FILE_CORR_OVERLAP_BA =   "corr_overlap_ba.parq"


def main():
    """Setup the parser and dispatch user input to the correct function."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Command: init
    parser_init = subparsers.add_parser('init')
    parser_init.add_argument(
        "-e", "--endpoint-first-events",
        help="path to phenotype file with first events (TSV)",
        required=True,
        type=Path
    )
    parser_init.add_argument(
        "-o", "--output",
        help="path to intermediate output file (Parquet)",
        required=True,
        type=Path
    )
    parser_init.set_defaults(func=init)

    # Command: compute
    parser_compute = subparsers.add_parser('compute')
    parser_compute.add_argument(
        "-i", "--input",
        help="path intermediate file (Parquet)",
        required=True,
        type=Path
    )
    parser_compute.add_argument(
        "-o", "--output-dir",
        help="path to output directory",
        required=True,
        type=Path
    )
    parser_compute.set_defaults(func=compute)

    # Command: inspect
    parser_inspect = subparsers.add_parser('inspect')
    parser_inspect.add_argument(
        "-a", "--endpoint-a",
        help="endpoint to compare (rows of contingency table)",
        required=True
    )
    parser_inspect.add_argument(
        "-b", "--endpoint-b",
        help="endpoint to compare (columns of contingency table)",
        required=True
    )
    parser_inspect.add_argument(
        "-i", "--input-dir",
        help="path to directory with input files",
        required=True,
        type=Path
    )
    parser_inspect.set_defaults(func=inspect)

    # Run argument parser
    args = parser.parse_args()
    if not vars(args):
        parser.print_help()
    else:
        args.func(args)


def init(args):
    """Extract necessary columns from the phenotype first event file.

    Here it was decided to not rely on pd.read_csv because:
    - it was slow
    - it was using at least 10x the amount of memory necessary and
      often resulting in getting OOM killed
    - we have control of the input file format, no need for Pandas to
      deal with edge cases

    This allows us to parse the input file with much less resources at
    the expense of more verbose code.
    """
    path_fevents = args.endpoint_first_events
    path_output = args.output

    with open(path_fevents) as f:
        # Get headers
        line = next(f).rstrip()
        headers = line.split(SEP)
        set_headers = set(headers)

        # Select only the endpoint columns
        endpoints = []
        pos_endpoints = set()
        for hh, header in enumerate(headers):
            has_age = header + "_AGE" in set_headers
            has_year = header + "_YEAR" in set_headers
            has_nevt = header + "_NEVT" in set_headers
            if has_age and has_year and has_nevt:
                endpoints.append(header)
                pos_endpoints.add(hh)

        # Parse the data into a numpy array.
        # We allocate a big enough array to fit all the rows from the
        # input file. It uses uint8 since we are only using the values
        # 0, 1 and 2 and we are concerned about memory usage.
        table = np.empty(
            shape=(MAX_LINES, len(endpoints)),
            dtype=np.uint8
        )
        for ll, line in enumerate(f):
            # Print out a friendly message if we are not able read all
            # of the input file.
            if ll >= MAX_LINES:
                raise RuntimeError(f"The input file `{path_fevents}` has more lines than MAX_LINES={MAX_LINES}. Please replace the value of MAX_LINES to be at least the number of lines of `{path_fevents}`. This can be checked with:\nwc -l {path_fevents}")

            # Display progress as we go so we know we are not stuck.
            if ll % 1000 == 0:
                print(f"at line: {ll}")

            # Process with parsing the line and putting it in the
            # numpy array.
            line = line.rstrip()
            row = line.split(SEP)
            row_cursor = 0
            for vv, value in enumerate(row):
                if vv in pos_endpoints:
                    uint8_val_excl = 2
                    x = uint8_val_excl if value == VAL_EXCL else value
                    table[ll, row_cursor] = x
                    row_cursor += 1

    # Trim the table
    final_line = ll + 1
    table = table[:final_line, :]

    # Use a pandas dataframe for named columns and Parquet output
    df = pd.DataFrame(
        data=table,
        columns=endpoints,
        copy=False
    )
    df.to_parquet(path_output)


def compute(args):
    """Compute contingency table and correlations for all endpoint combinations.

    While we used uint8 for the input file, we now require float64
    since the operations on the matrices would trigger integer
    overflow.
    """
    df = pd.read_parquet(args.input)

    # Use a numpy array to do computations with vectorized operations
    arr = df.to_numpy()

    # Assign indivuduals as cases (1), controls (0), or excluded controls (2)
    val_case = 1
    val_control = 0
    val_excl = 2
    cases = (arr == val_case).astype(np.float64)
    controls = (arr == val_control).astype(np.float64)
    excl = (arr == val_excl).astype(np.float64)

    # Compute counts
    n_cases_cases = cases.T.dot(cases)
    n_controls_controls = controls.T.dot(controls)
    n_excl_excl = excl.T.dot(excl)

    n_cases_controls = cases.T.dot(controls)
    n_controls_cases = n_cases_controls.T

    n_cases_excl = cases.T.dot(excl)
    n_excl_cases = n_cases_excl.T

    n_controls_excl = controls.T.dot(excl)
    n_excl_controls = n_controls_excl.T

    # Compute correlation coefficients for all endpoints
    fphi, fchisq, fcase_ratio, overlap_ab, overlap_ba = correlations(
        n_cases_cases, n_cases_controls, n_cases_excl,
        n_controls_cases, n_controls_controls, n_controls_excl,
        n_excl_cases, n_excl_controls, n_excl_excl
    )

    # Output files
    endpoints = df.columns
    outputs = [
        (n_cases_cases, FILE_CASES_CASES),
        (n_cases_controls, FILE_CASES_CONTROLS),
        (n_cases_excl, FILE_CASES_EXCL),
        (n_controls_cases, FILE_CONTROLS_CASES),
        (n_controls_controls, FILE_CONTROLS_CONTROLS),
        (n_controls_excl, FILE_CONTROLS_EXCL),
        (n_excl_cases, FILE_EXCL_CASES),
        (n_excl_controls, FILE_EXCL_CONTROLS),
        (n_excl_excl, FILE_EXCL_EXCL),
        (fphi, FILE_CORR_PHI),
        (fchisq, FILE_CORR_CHISQ),
        (fcase_ratio, FILE_CORR_CASE_RATIO),
        (overlap_ab, FILE_CORR_OVERLAP_AB),
        (overlap_ba, FILE_CORR_OVERLAP_BA)
    ]
    for df, output_path in outputs:
        pd.DataFrame(
            data=df,
            columns=endpoints,
            index=endpoints,
            copy=False
        ).to_parquet(args.output_dir / output_path)



def correlations(
        n11,  n10,  n12,
        n01,  n00, _n02,
        n21, _n20, _n22
):
    """Swiftly compute correlation coefficients using operations on matrices"""
    # Notation:
    #                    cases  controls  excluded controls
    #             cases  n11    n10       n12
    #          controls  n01    n00       n02
    # excluded controls  n21    n20       n22
    phi = (
        (n11 * n00 - n10 * n01)
        / np.sqrt(
            (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
        )
    )

    chisq = (n11 + n10 + n01 + n00) * phi * phi

    case_ratio = n11 / (n10 + n01 + n11 + n21 + n12)

    overlap_ab = n11 / (n11 + n01 + n21)
    overlap_ba = n11 / (n11 + n10 + n12)

    return phi, chisq, case_ratio, overlap_ab, overlap_ba


def inspect(args):
    """Get the contingency table and correlations for 2 endpoints"""
    def lookup(path):
        return pd.read_parquet(path)[col][row]

    row = args.endpoint_a
    col = args.endpoint_b

    n11 = args.input_dir / FILE_CASES_CASES
    n10 = args.input_dir / FILE_CASES_CONTROLS
    n12 = args.input_dir / FILE_CASES_EXCL
    n01 = args.input_dir / FILE_CONTROLS_CASES
    n00 = args.input_dir / FILE_CONTROLS_CONTROLS
    n02 = args.input_dir / FILE_CONTROLS_EXCL
    n21 = args.input_dir / FILE_EXCL_CASES
    n20 = args.input_dir / FILE_EXCL_CONTROLS
    n22 = args.input_dir / FILE_EXCL_EXCL
    phi = args.input_dir / FILE_CORR_PHI
    chisq = args.input_dir / FILE_CORR_CHISQ
    case_ratio = args.input_dir / FILE_CORR_CASE_RATIO
    overlap_ab = args.input_dir / FILE_CORR_OVERLAP_AB
    overlap_ba = args.input_dir / FILE_CORR_OVERLAP_BA

    table = np.array([
        [lookup(n11), lookup(n10), lookup(n12)],
        [lookup(n01), lookup(n00), lookup(n02)],
        [lookup(n21), lookup(n20), lookup(n22)]
    ], dtype=np.int32)
    print(f"{row} \\ {col}\n{table}")

    print(f"φ: {lookup(phi)}")
    print(f"χ²: {lookup(chisq)}")
    print(f"case ratio: {lookup(case_ratio)}")

    perc_overlap_ab = lookup(overlap_ab) * 100
    perc_overlap_ba = lookup(overlap_ba) * 100
    print(f"cases overlap: % of {args.endpoint_a} in {args.endpoint_b}: {perc_overlap_ab:6.2f}%")
    print(f"cases overlap: % of {args.endpoint_b} in {args.endpoint_a}: {perc_overlap_ba:6.2f}%")


if __name__ == '__main__':
    main()

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


# -- Configuration

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

# Define treshold for individual-level data
MIN_CASES = 5

# Output files
FILE_CASES_CASES        = "n_cases_cases.parq"
FILE_CASES_CONTROLS     = "n_cases_controls.parq"
FILE_CASES_EXCL         = "n_cases_excl.parq"
FILE_CONTROLS_CASES     = "n_controls_cases.parq"
FILE_CONTROLS_CONTROLS  = "n_controls_controls.parq"
FILE_CONTROLS_EXCL      = "n_controls_excl.parq"
FILE_EXCL_CASES         = "n_excl_cases.parq"
FILE_EXCL_CONTROLS      = "n_excl_controls.parq"
FILE_EXCL_EXCL          = "n_excl_excl.parq"
FILE_CORR_PHI           = "corr_phi.parq"
FILE_CORR_CHISQ         = "corr_chisq.parq"
FILE_CORR_JACCARD_INDEX = "corr_jaccard_index.parq"
FILE_CORR_SHARED_OF_A   = "corr_shared_of_a.parq"
FILE_CORR_SHARED_OF_B   = "corr_shared_of_b.parq"


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
    parser_init.add_argument(
        "-d", "--delimiter",
        help="input file field seperator (e.g. '\\t' or ',')",
        required=False,
        default="\t"
    )
    parser_init.add_argument(
        "-x", "--value-excluded-control",
        help="encoding of excluded controls in input file (e.g. 'NA' or '')",
        required=False,
        default="NA"
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

    # Command: csv
    parser_csv = subparsers.add_parser('csv')
    parser_csv.add_argument(
        "-i", "--input-dir",
        help="path to directory with input files",
        required=True,
        type=Path
    )
    parser_csv.add_argument(
        "-o", "--csv-output",
        help="CSV output path",
        required=True,
        type=Path
    )
    parser_csv.add_argument(
        "-k", "--keep-all",
        help="keep all data, even individual-level data, in the CSV output",
        action='store_true'
    )
    parser_csv.set_defaults(func=to_csv)

    # Run argument parser
    args = parser.parse_args()

    # Unescape the delimiter, as the shell escapes the \
    if args.func == init and args.delimiter == "\\t":
        args.delimiter = "\t"

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
        headers = line.split(args.delimiter)
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
            row = line.split(args.delimiter)
            row_cursor = 0
            for vv, value in enumerate(row):
                if vv in pos_endpoints:
                    uint8_val_excl = 2
                    x = uint8_val_excl if value == args.value_excluded_control else value
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
    fphi, fchisq, fjaccard_index, shared_of_a, shared_of_b = correlations(
        n_cases_cases,    n_cases_controls,    n_cases_excl,
        n_controls_cases, n_controls_controls, n_controls_excl,
        n_excl_cases,     n_excl_controls,     n_excl_excl
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
        (fjaccard_index, FILE_CORR_JACCARD_INDEX),
        (shared_of_a, FILE_CORR_SHARED_OF_A),
        (shared_of_b, FILE_CORR_SHARED_OF_B)
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
    #                      B cases  B controls  B excluded controls
    #             A cases  n11      n10         n12
    #          A controls  n01      n00         n02
    # A excluded controls  n21      n20         n22
    phi = (
        (n11 * n00 - n10 * n01)
        / np.sqrt(
            (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
        )
    )

    chisq = (n11 + n10 + n01 + n00) * phi * phi

    jaccard_index = n11 / (n10 + n01 + n11 + n21 + n12)

    shared_of_a = n11 / (n11 + n10 + n12)
    shared_of_b = n11 / (n11 + n01 + n21)

    return phi, chisq, jaccard_index, shared_of_a, shared_of_b


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
    jaccard_index = args.input_dir / FILE_CORR_JACCARD_INDEX
    shared_of_a = args.input_dir / FILE_CORR_SHARED_OF_A
    shared_of_b = args.input_dir / FILE_CORR_SHARED_OF_B

    table = np.array([
        [lookup(n11), lookup(n10), lookup(n12)],
        [lookup(n01), lookup(n00), lookup(n02)],
        [lookup(n21), lookup(n20), lookup(n22)]
    ], dtype=np.int32)
    print(f"(A) {row} \\ (B) {col}\n{table}")

    print(f"φ: {lookup(phi)}")
    print(f"χ²: {lookup(chisq)}")
    print(f"Jaccard index: {lookup(jaccard_index)}")

    perc_shared_of_a = lookup(shared_of_a) * 100
    perc_shared_of_b = lookup(shared_of_b) * 100
    print(f"{perc_shared_of_a:6.2f}% of {args.endpoint_a} cases are shared cases with {args.endpoint_b}")
    print(f"{perc_shared_of_b:6.2f}% of {args.endpoint_b} cases are shared cases with {args.endpoint_a}")


def to_csv(args):
    """Output correlations of interest for all endpoints into a CSV file"""
    # Load correlation files
    # NOTE(vincent): counts files used by the following correlations
    # *MUST BE* loaded to be checked for individual-level data.
    df_jaccard_index = pd.read_parquet(args.input_dir / FILE_CORR_JACCARD_INDEX)
    df_cases_cases = pd.read_parquet(args.input_dir / FILE_CASES_CASES)
    df_shared_of_a = pd.read_parquet(args.input_dir / FILE_CORR_SHARED_OF_A)
    df_shared_of_b = pd.read_parquet(args.input_dir / FILE_CORR_SHARED_OF_B)

    # Load count files.
    # These are used to check that the underlying correlations don't
    # use any individual-level data.
    # It is important to check the counts even if they are not present
    # in the output, as someone with partial information could
    # traceback individual-level data.
    # For example, someone knowing that:
    # - there are 10 cases for endpoint A
    # - shared cases of A is 0.2
    # - shared cases of B is 1.0
    # can deduce they are 2 cases (individual-level data) for endpoint B.
    if not args.keep_all:
        df_cases_cases = pd.read_parquet(args.input_dir / FILE_CASES_CASES)
        df_cases_controls = pd.read_parquet(args.input_dir /FILE_CASES_CONTROLS)
        df_controls_cases = pd.read_parquet(args.input_dir /FILE_CONTROLS_CASES)
        df_excl_cases = pd.read_parquet(args.input_dir /FILE_EXCL_CASES)
        df_cases_excl = pd.read_parquet(args.input_dir /FILE_CASES_EXCL)

        # Matrix (symmetrical) that tracks wich (endp A, endp B) can
        # be kept in the output.
        df_keep = (
            ((df_cases_cases > MIN_CASES) | (df_cases_cases == 0))
            & ((df_cases_controls > MIN_CASES) | (df_cases_controls == 0))
            & ((df_controls_cases > MIN_CASES) | (df_controls_cases == 0))
            & ((df_excl_cases > MIN_CASES) | (df_excl_cases == 0))
            & ((df_cases_excl > MIN_CASES) | (df_cases_excl == 0))
        )

    with open(args.csv_output, "w", newline="") as output:
        csv_writer = csv.writer(output)
        # Add headers to CSV output
        csv_writer.writerow([
            "endpoint_a",
            "endpoint_b",
            "jaccard_index",
            "case_overlap_N",
            "ratio_shared_of_a",
            "ratio_shared_of_b"
        ])

        for endp_b, indexes in df_jaccard_index.items():
            for endp_a, jaccard_index in indexes.items():
                # Our matrix representation has endp_a as the "row endpoint"
                # and endp_b as the "column endpoint".
                # However, pandas DataFrames are indexed first by column, and
                # then by row. That's why we use [endp_b][endp_a] here.
                # This matters because the "df_share_of" are not symmetricals.

                # Check for individual-level data
                if args.keep_all:
                    keep_row = True
                else:
                    keep_row = df_keep[endp_b][endp_a]

                if keep_row:
                    # Here we assume the set of endpoints are exactly
                    # the same for: case overlap, shared of A and shared
                    # of B. So we assume indexing by endp_b and endp_a
                    # will always succeed.
                    shared_of_a = df_shared_of_a[endp_b][endp_a]
                    shared_of_b = df_shared_of_b[endp_b][endp_a]
                    overlap_N = int(df_cases_cases[endp_b][endp_a])

                    csv_writer.writerow([
                        endp_a,
                        endp_b,
                        jaccard_index,
                        overlap_N,
                        shared_of_a,
                        shared_of_b
                    ])


if __name__ == '__main__':
    main()

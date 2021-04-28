# NOTES:
# - order of cases/controls matters
# - how to output?

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    print(datetime.now(), "Init")
    # CLI argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f", "--fg-cases-controls",
        help="path to FinnGen endpoint cases and controls (Parquet)",
        required=True,
        type=Path
    )
    parser.add_argument(
        "-c", "--custom-cases-controls",
        required=True,
        type=Path
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path
    )

    args = parser.parse_args()


    # FG: Load data
    print(datetime.now(), "(FG) load data")
    arr_fg = load_fg_data(args.fg_cases_controls)

    # FG: Assign cases and controls
    print(datetime.now(), "(FG) Assign cases and controls ")
    fg_cases, fg_controls = extract_cases_controls(arr_fg)

    # Custom: load data
    print(datetime.now(), "(Custom) load data")
    arr_custom = load_custom_data(args.custom_cases_controls)

    # Custom: assign cases and controls
    print(datetime.now(), "(Custom) Assign cases and controls ")
    custom_cases, custom_controls = extract_cases_controls(arr_custom)

    # Compute counts
    print(datetime.now(), "Compute counts ")
    (
        n_cases_cases,    n_cases_controls,
        n_controls_cases, n_controls_controls
    ) = counts(custom_cases, custom_controls, fg_cases, fg_controls)

    # Compute correlation
    print(datetime.now(), "Compute correlation")
    corr = correlation(
        n_cases_cases,    n_cases_controls,
        n_controls_cases, n_controls_controls
    )

    # Output
    print(datetime.now(), "Output")
    print(corr)


def load_fg_data(fg_cases_controls):
    arr_fg = pd.read_parquet(fg_cases_controls).to_numpy()
    return arr_fg


def load_custom_data(custom_cases_controls):
    df_custom = pd.read_csv(custom_cases_controls, usecols=["FINNGENID", "CUSTOM_ENDPOINT"])
    # TODO check FINNGENID order
    arr_custom = df_custom.to_numpy()
    return arr_custom


def extract_cases_controls(arr):
    val_case = 1

    cases = (arr == val_case).astype(np.float64)
    controls = (arr != val_case).astype(np.float64)

    return cases, controls


def counts(custom_cases, custom_controls, fg_cases, fg_controls):
    n_cases_cases = custom_cases.T.dot(fg_cases)
    n_cases_controls = custom_cases.T.dot(fg_controls)
    n_controls_cases = custom_controls.T.dot(fg_cases)
    n_controls_controls = custom_controls.T.dot(fg_controls)

    return (
        n_cases_cases,    n_cases_controls,
        n_controls_cases, n_controls_controls
    )


def correlation(
        n11, n10,
        n01, n00
):
    case_control_overlap = n11 / (n11 + n10 + n01 + n00)
    return case_control_overlap


if __name__ == '__main__':
    main()

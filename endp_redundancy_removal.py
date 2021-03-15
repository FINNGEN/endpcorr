import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--threshold-case-ratio",
        help="endpoints with a case ratio greater than this value are deemed redundant",
        required=True,
        type=float
    )
    parser.add_argument(
        "-g", "--gwas-sig-hits",
        help="path to file with GWAS significant hits for each endpoint (CSV)",
        required=True,
        type=Path
    )
    parser.add_argument(
        "-c", "--cases-controls-excl",
        help="path to file with cases / controls / excluded controls indicator (Parquet)",
        required=True,
        type=Path
    )
    parser.add_argument(
        "-r", "--case-ratios",
        help="path to file with precomputed case ratios (Parquet)",
        required=True,
        type=Path
    )
    parser.add_argument(
        "-o", "--output",
        help="path to output file (CSV)",
        required=True,
        type=Path
    )
    parser.add_argument(
        "--debug",
        help="print debugging information",
        action="store_true"
    )
    args = parser.parse_args()

    # Load file with GWsig hits
    df_gwsig = pd.read_csv(args.gwas_sig_hits)

    # Load file with cases
    df_cases_controls_excl = pd.read_parquet(args.cases_controls_excl)
    val_case = 1
    ser_cases = (df_cases_controls_excl == val_case).sum()
    df_cases = pd.DataFrame({
        "endpoint": ser_cases.index,
        "cases": ser_cases.values
    })

    # Load file with case ratios
    df_case_ratio = pd.read_parquet(args.case_ratios)

    # Add info on N cases
    df_gwsig = df_gwsig.merge(df_cases, on="endpoint")

    # Sort
    df_gwsig = df_gwsig.sort_values(
        by=["gwas_sig_hits", "cases"],
        ascending=[False, True],
    )

    # Alg step
    df = df_gwsig.copy()
    df_accepted = pd.DataFrame({
        "endpoint": [],
        "redundant_endpoints_removed": []
    })
    while df.shape[0] > 0:
        df, df_accepted = step(
            args.threshold_case_ratio,
            df,
            df_case_ratio,
            df_accepted,
            args.debug
        )

    # Output
    df_accepted.merge(
        df_gwsig,
        on="endpoint"
    ).loc[
        :,
        # Reorder columns
        ["endpoint", "gwas_sig_hits", "cases", "redundant_endpoints_removed"]
    ].to_csv(args.output, index=False)


def step(
        cr_threshold,
        df_gwsig,
        df_case_ratio,
        df_accepted,
        debug
):
    # Select "accepted" endpoint
    top_row = df_gwsig.iloc[0]
    endp = top_row.endpoint

    # Remove accepted endpoint from the list of endpoints to do
    df_gwsig = df_gwsig.iloc[1:]

    # Add case-ratio info wrt to accepted endpoint
    endp_cr = df_case_ratio.loc[:, [endp]]
    endp_cr = endp_cr.rename(columns={endp: "case_ratio"})
    endp_cr["endpoint"] = endp_cr.index

    # Find redundant endpoints based on case-ratio threshold
    df = df_gwsig.merge(endp_cr, on="endpoint")
    df_redundant = df.loc[df.case_ratio > cr_threshold, :]
    df = df.loc[df.case_ratio <= cr_threshold, :]

    if debug and df_redundant.shape[0] > 0:
        print(f"Current endpoint: {endp}")
        print(f"redundant endpoints: {df_redundant.shape[0]}")
        print(df_redundant, end="\n\n")

    # Keep same shape as the input
    df_gwsig = df.drop("case_ratio", axis="columns")

    # Add info on accepted endpoint
    df_accepted = df_accepted.append({
            "endpoint": endp,
            "redundant_endpoints_removed": " ".join(df_redundant.endpoint.values)
        },
        ignore_index=True
    )
    return df_gwsig, df_accepted


if __name__ == '__main__':
    main()

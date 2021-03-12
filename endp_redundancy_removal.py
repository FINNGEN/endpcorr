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
    accepted = []
    df = df_gwsig.copy()
    while df.shape[0] > 0:
        df, accepted = step(
            df,
            df_case_ratio,
            accepted,
            args.threshold_case_ratio,
            args.debug
        )

    # Output
    df_accepted = pd.DataFrame({"endpoint": accepted})
    df_accepted.merge(
        df_gwsig.loc[:, ["endpoint", "gwas_sig_hits"]],
        on="endpoint"
    ).to_csv(args.output, index=False)



def step(df_gwsig, df_case_ratio, accepted, cr_threshold, debug):
    top_row = df_gwsig.iloc[0]
    df_gwsig = df_gwsig.iloc[1:]

    endp = top_row.endpoint
    accepted.append(endp)

    endp_cr = df_case_ratio.loc[:, [endp]]
    endp_cr = endp_cr.rename(columns={endp: "case_ratio"})
    endp_cr["endpoint"] = endp_cr.index

    df = df_gwsig.merge(endp_cr, on="endpoint")
    if debug:
        discarded = df.loc[df.case_ratio > cr_threshold, :]
    df = df.loc[df.case_ratio <= cr_threshold, :]

    if debug and discarded.shape[0] > 0:
        print(f"Current endpoint: {endp}")
        print(f"discarding {discarded.shape[0]} endpoints:")
        print(discarded, end="\n\n")

    df_gwsig = df.drop("case_ratio", axis="columns")
    return df_gwsig, accepted


if __name__ == '__main__':
    main()

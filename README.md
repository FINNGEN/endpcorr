Usage:


1. Convert the wide- first-event endpoint file (R13v2.0) to a zstd-compressed CSV:
  ```sh
pv finngen_R13_endpoint_2.0.txt.gz | zcat | xsv fmt -d"\t" | zstd -T0 -q -o ~/data/r13_mine/data/finngen_R13_endpoint_2.0.csv.zst
  ```

2. Run the Endpcorr init:
  ```sh
time uv run --python 3.13 --with numpy,pandas,pyarrow main.py init -e <(zstdcat ~/data/r13_mine/data/finngen_R13_endpoint_2.0.csv.zst)  -o ~/data/r13.0_results/correlations/intermediate__r13__2026-07-03.parquet -d ',' -x NA
  ```

#!/usr/bin/env python

import argparse

from hack.transistors.transistors import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Commandline interface for KBC for transistors."
    )

    parser.add_argument(
        "--stg-temp-min",
        help="Extract minimum storage temperature.",
        action="store_true",
    )
    parser.add_argument(
        "--stg-temp-max",
        help="Extract maximum storage temperature.",
        action="store_true",
    )
    parser.add_argument("--polarity", action="store_true", help="Extract polarity.")
    parser.add_argument(
        "--ce-v-max",
        help="Extract maximum collector-emitter voltage.",
        action="store_true",
    )
    parser.add_argument("--parse", action="store_true", help="Parse the dataset.")
    parser.add_argument(
        "--first-time",
        help="Run all stages of the pipeline assuming a parsed dataset.",
        action="store_true",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=float("inf"),
        help="The max number of docs to parse from each dataset (dev/test/train).",
    )
    parser.add_argument(
        "--parallel", type=int, default=8, help="Set the level of parallelization."
    )
    parser.add_argument("--gpu", type=str, help="Use the specified GPU index.")
    parser.add_argument(
        "--conn-string", type=str, help="Connection string to the PosgreSQL Database."
    )
    parser.add_argument("--log-dir", type=str, help="Directory to output log files.")
    parser.add_argument("-v", help="Set INFO level logging.", action="store_true")

    args = parser.parse_args()

    main(
        args.conn_string,
        stg_temp_min=args.stg_temp_min,
        stg_temp_max=args.stg_temp_max,
        polarity=args.polarity,
        ce_v_max=args.ce_v_max,
        max_docs=args.max_docs,
        parse=args.parse,
        first_time=args.first_time,
        gpu=args.gpu,
        parallel=args.parallel,
        log_dir=args.log_dir,
        verbose=args.v,
    )

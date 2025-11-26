""""
Command-line interface for TLE propagator.
"""
import argparse
import sys
from pathlib import Path

from .tle_propagator import Config, TLEPropagator
# TODO verbose, config YAML, integrator/force model selection

def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tle-propagator", description="TLE Propagator")
    
    # TLE source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input_file", type=str, help="Input file with TLE data")
    input_group.add_argument("--norad_id",         type=str, help="NORAD ID of the satellite")
    # Propagation parameters
    parser.add_argument("--start_time", type=float, required=True, help="Start time in seconds since TLE epoch")
    parser.add_argument("--end_time",   type=float, required=True, help="End time in seconds since TLE epoch")
    parser.add_argument("--time_step",  type=float, required=True, help="Time step in seconds")
    # Output parameters
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--plot",             action="store_true",        help="Generate plots")
    parser.add_argument("--yaml",             action ="store_true",       help="Generate YAML answer sheet")
    
    return parser

def main() -> None:
    parser = cli_parser()
    args = parser.parse_args()

    # Select input source
    input_file = Path()
    if args.input_file is not None:
        input_file = Path(args.input_file)
        if not input_file.exists() or not input_file.is_file():
            print(f"Error: Input file {input_file} does not exist or is not a file.", file=sys.stderr)
            sys.exit(1)
    elif args.norad_id is not None:
        from .tle_retriever import retrieve_tle
        try:
            input_file = retrieve_tle(args.norad_id)
        except Exception as e:
            print(f"Error: could not retrieve TLE data. {e}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tle_propagator = TLEPropagator(
        config=Config(
            input_file=input_file,
            output_dir=output_dir,
            times=(args.start_time, args.end_time, args.time_step),
            plot=args.plot,
            yaml=args.yaml
        )
    )
    try:
        tle_propagator.run()
    except Exception as e:
        raise RuntimeError("Failure in TLE propagation.") from e

if __name__ == "__main__":
    main()
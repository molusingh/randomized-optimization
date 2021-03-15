#!/usr/bin/env python

from randomized_optimization import run_problem_config
import json
import os
import sys
import argparse

NQUEENS_PATH = "nqueens_config.json"
FLIP_FLOP_PATH = "flipflop_config.json"
FOUR_PEAKS_PATH = "fourpeaks_config.json"
NN_PATH = "nnconfig.json"

def main(paths, output):
    for path in paths:
        with open(path, 'r') as fp:
            config = json.load(fp)
        results = run_problem_config(config, output)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Execute Randomized Optimization experiments')
    my_parser.add_argument('--path',type=str,help='the path to configuration file, if not provided runs all default problems')
    my_parser.add_argument('--output',type=str,help='output directory, default is output', default="output")
    args = my_parser.parse_args()
    paths = [NQUEENS_PATH, FLIP_FLOP_PATH, FOUR_PEAKS_PATH, NN_PATH]
    if args.path:
        paths = [args.path]
    main(paths, args.output)
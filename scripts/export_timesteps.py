"""Export a descending schedule from a Stage-I Pareto front generated locally."""
import argparse
import ast
import math
from pathlib import Path
import re

LINE = re.compile(r"^\s*(\[[^\]]+\]|\([^\)]+\))\s+fid=(\S+)\s+lat=(\S+)\s+std=(\S+)\s*$")

def parse_line(line):
    match = LINE.match(line)
    if match is None:
        return None
    schedule = list(ast.literal_eval(match.group(1)))
    objectives = tuple(float(match.group(i)) for i in (2, 3, 4))
    return schedule, objectives

def select_schedule(rows):
    if not rows:
        raise ValueError("No Pareto candidates were parsed")
    valid = []
    for schedule, objectives in rows:
        if not schedule or any(type(t) is not int or not 0 <= t < 1000 for t in schedule):
            raise ValueError("Schedules must contain integer indices in [0, 999]")
        if len(set(schedule)) != len(schedule):
            raise ValueError("A schedule contains duplicate timestep indices")
        if len(objectives) != 3 or not all(math.isfinite(x) for x in objectives):
            continue
        valid.append((schedule, objectives))
    if not valid:
        raise ValueError("No candidates with three finite objectives")
    minima = [min(row[1][i] for row in valid) for i in range(3)]
    maxima = [max(row[1][i] for row in valid) for i in range(3)]
    def distance(row):
        return sum(((row[1][i] - minima[i]) / (maxima[i] - minima[i])) ** 2
                   if maxima[i] > minima[i] else 0.0 for i in range(3))
    return sorted(min(valid, key=distance)[0], reverse=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [row for line in args.front.read_text(encoding="utf-8").splitlines()
            if (row := parse_line(line)) is not None]
    schedule = select_schedule(rows)
    import numpy as np
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.asarray(schedule, dtype=np.int64))
    print(f"Exported {len(schedule)} timesteps to {args.output}")

if __name__ == "__main__":
    main()

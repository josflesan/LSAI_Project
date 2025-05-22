import re
import pandas as pd
from typing import List

PATTERN = re.compile(r"Step:\s*(\d+)\s*\|\s*Loss:\s*([\d.]+)")


def parse_file(filename: str, run: str) -> pd.DataFrame:
    """Parses a log output file into a dataframe for plotting"""

    seen = set()
    data = {"step": [], "loss": [], "type": []}

    with open(filename, "r") as f:
        for line in f:
            match = PATTERN.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))

                if step in seen:
                    continue

                data["step"].append(step)
                data["loss"].append(loss)
                seen.add(step)

    # Add categorical column
    data["type"] = [run] * len(data["step"])

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Sort
    df = df.sort_values(by="step").reset_index(drop=True)

    return df


def parse_all(filenames: List[str], runs: List[str]) -> pd.DataFrame:
    """Parses all of the log files into a single dataframe"""

    # Initialize dataframe
    final_df = pd.DataFrame(columns=["step", "loss", "type"])

    for file, run in zip(filenames, runs):
        df = parse_file(file, run)
        final_df = pd.concat([final_df, df], axis=0, ignore_index=True)

    return final_df

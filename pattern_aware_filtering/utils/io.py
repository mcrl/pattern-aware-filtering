import os


def read_txt(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]
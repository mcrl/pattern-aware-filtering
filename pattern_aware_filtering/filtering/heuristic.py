import os
import re
import json




allowed_chars = set([".", "?","!", "\"", "'"])  # allowed characters for nopunc filtering

def nopunc_filtering(line):
    """Returns True if the line should be removed"""
    line = line.lower().strip()
    last_char = line[-1] if len(line) > 0 else ""
    if last_char in allowed_chars:
        return False
    return True

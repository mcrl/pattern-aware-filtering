import re
from functools import partial

BP = 1
NORM = 0
RED = 1000
GREEN = 1
DEFAULT_PREDICTION = BP

MAIN_PATTERNS = [r"g{2,}", r"g{2,}(y+g{1,})+g",  r"g{2,}([yr]{0,3}g{1,})+g"]
MAIN_PATTERNS_COMPILED = [re.compile(pttn) for pttn in MAIN_PATTERNS]

def color_func(count, red=RED, green=GREEN):
    return "r" if count > red else "y" if count > green else "g"

def override_color(lines, colors):
    indices = [i for i, line in enumerate(lines) if line.strip() in ["{", "}", ""]]
    for idx in indices:
        colors[idx] = "y"
    return colors

def ld_extractor_engine(lines, counts, threshold=1):
    predictions = [BP if count > threshold else NORM for count in counts]
    return predictions
    
def ld_extractor(lines, counts, threshold=1):
    predictions = ld_extractor_engine(lines, counts, threshold)
    result = [lines[i] for i, pred in enumerate(predictions) if pred == NORM]
    return result

def apply_patterns(color, pttns, mode, res):
    for pttn in pttns:
        matches = re.finditer(pttn, color)
        for match in matches:
            start, end = match.span()
            res[start:end] = [mode]*(end-start)

def pld_engine(lines, counts, red=RED, green=GREEN, main_patterns_compiled=MAIN_PATTERNS_COMPILED):
    DEFAULT_PREDICTION = BP
    line_count = len(counts)
    color_func_setting = partial(color_func, red=red, green=green)
    colors = [color_func_setting(count) for count in counts]
    colors = override_color(lines, colors)
    codes = "".join(colors)
    res = [DEFAULT_PREDICTION]*line_count
    apply_patterns(codes, main_patterns_compiled, NORM, res)

    return res

def pld_extractor(lines, counts, red=RED, green=GREEN,  main_patterns_compiled=MAIN_PATTERNS_COMPILED):
    res = pld_engine(lines, counts, red, green,  main_patterns_compiled)
    result = [lines[i] for i, mode in enumerate(res) if mode == NORM]
    return result

punc_pattern_main_text_pttn = [r"g+",  r"g+([yr]{0,3}g+)+"]
punc_pattern_main_text_pttn_compiled = [re.compile(pttn) for pttn in punc_pattern_main_text_pttn]


allowed_last_chars = set([".", "?", "!", "\"", "'"])

def ptf_engine(lines, pattern_compiled=punc_pattern_main_text_pttn_compiled):
    def _line_function(line):
        last_char = line.strip()[-1] if line.strip() else ""
        return last_char in allowed_last_chars
    colors = ['g' if _line_function(line) else 'y' for line in lines]
    codes = "".join(colors)
    res = [BP] * len(codes)
    apply_patterns(codes, pattern_compiled, NORM, res)
    return res


def ptf_extractor(lines, pattern_compiled=punc_pattern_main_text_pttn_compiled):
    res = ptf_engine(lines, pattern_compiled)
    result = [lines[i] for i, mode in enumerate(res) if mode == NORM]
    return result
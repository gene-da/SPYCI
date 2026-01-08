import re
from typing import Union

import re
from typing import Union

METRIC_SUFFIXES = [
    ('T', 1e12),
    ('G', 1e9),
    ('Meg', 1e6),
    ('K', 1e3),
    ('mil', 25.4e-6),
    ('m', 1e-3),
    ('u', 1e-6),
    ('n', 1e-9),
    ('p', 1e-12),
    ('f', 1e-15),
    ('a', 1e-18),
]

# For parsing metric suffixes case-sensitively
PARSE_SUFFIX_MAP = {
    'T': 1e12,
    'G': 1e9,
    'Meg': 1e6,
    'M': 1e6,    # Acceptable shorthand
    'K': 1e3,
    'k': 1e3,
    'mil': 25.4e-6,
    'm': 1e-3,
    'u': 1e-6,
    'n': 1e-9,
    'p': 1e-12,
    'f': 1e-15,
    'a': 1e-18,
}

def normalize_value(val: Union[str, float, int]) -> float:
    """Convert a SPICE metric string or number to float."""
    if isinstance(val, (int, float)):
        return float(val)
    
    val = val.strip()
    match = re.match(r'^([+-]?[\d.]+(?:[eE][+-]?\d+)?)([a-zA-Z]+)?$', val)
    if not match:
        raise ValueError(f"Invalid SPICE format: '{val}'")

    num_part = float(match.group(1))
    suffix = match.group(2)

    if not suffix:
        return num_part

    suffix = suffix.strip()

    # Normalize special cases
    if suffix.lower() == 'meg':
        return num_part * PARSE_SUFFIX_MAP['Meg']
    if suffix.lower() == 'mil':
        return num_part * PARSE_SUFFIX_MAP['mil']
    
    if suffix in PARSE_SUFFIX_MAP:
        return num_part * PARSE_SUFFIX_MAP[suffix]

    raise ValueError(f"Unknown metric suffix '{suffix}' in '{val}'")

def format_value(val: Union[str, float, int], precision: int = 6) -> str:
    """Return the value formatted as canonical SPICE metric string."""
    value = normalize_value(val)

    # Explicit zero: always format as "0"
    if value == 0:
        return "0"

    for suffix, factor in METRIC_SUFFIXES:
        scaled = value / factor
        if suffix == 'mil':
            # Must match exactly to use mil
            if abs(value - factor * round(scaled)) < 1e-9:
                return f"{int(round(scaled))}{suffix}"
        elif 0.1 <= abs(scaled) < 1e3:
            return f"{round(scaled, precision):g}{suffix}"

    return f"{value:.{precision}g}"

if __name__ == '__main__':
    print(format_value(1000000))    # → 1000000.0
    print(format_value("1M"))       # → 1e6
    print(format_value("1Meg"))      # → 1e6
    print(format_value("1m"))        # → 1e-3
    print(format_value("25mil"))     # → 0.000635
    print(format_value("4.7k"))      # → 4700.0
    print(format_value("10u"))       # → 1e-5
    print(format_value("3.3"))       # → 3.3
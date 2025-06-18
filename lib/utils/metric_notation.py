import numpy as np
import re

def fm(s):
    multipliers = {
        'y': 1e-24, 'z': 1e-21, 'a': 1e-18, 'f': 1e-15,
        'p': 1e-12, 'n': 1e-9,  'u': 1e-6,  'µ': 1e-6,
        'm': 1e-3,  '': 1,      'k': 1e3,   'K': 1e3,
        'M': 1e6,   'G': 1e9,   'T': 1e12,  'P': 1e15,
        'E': 1e18,  'Z': 1e21,  'Y': 1e24
    }

    # Handle native and numpy numeric types
    if isinstance(s, (int, float, np.number)):
        return float(s)

    s = s.strip()
    match = re.fullmatch(r'(-?\d*\.?\d*)([a-zA-Zµ]?)', s)
    if not match:
        raise ValueError(f"Invalid metric string: '{s}'")

    num_str, prefix = match.groups()
    return float(num_str) * multipliers.get(prefix, 1)

def tm(value, precision=2):
    if value == 0:
        return "0"

    prefixes = {
        -24: 'y', -21: 'z', -18: 'a', -15: 'f',
        -12: 'p', -9: 'n', -6: 'µ', -3: 'm',
         0: '',   3: 'k',  6: 'M',  9: 'G',
        12: 'T', 15: 'P', 18: 'E', 21: 'Z', 24: 'Y'
    }

    import math

    exponent = int(math.floor(math.log10(abs(value)) // 3 * 3))
    exponent = max(min(exponent, 24), -24)

    scaled = value / (10 ** exponent)
    prefix = prefixes.get(exponent, f'e{exponent}')

    # If scaled is effectively an integer, round to int
    if round(scaled, precision).is_integer():
        return f"{int(round(scaled))}{prefix}"
    else:
        return f"{round(scaled, precision)}{prefix}"

def needs_conversion(s):
    import re
    s = s.strip()
    return bool(re.fullmatch(r'-?\d*\.?\d*[a-zA-Zµ]?', s)) and not s.replace('.', '', 1).lstrip('-').isdigit()
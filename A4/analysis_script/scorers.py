import re

#--------------------------------------------------------------------------------
# Potentially useful helper (Karpathy)
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(assistant_response: str) -> str:
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(assistant_response)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

#--------------------------------------------------------------------------------
# Potentially useful metrics

def has_useful_reasoning(assistant_response: str) -> bool:
    """Returns whether the assistant response contains at least one equation which supports the final result"""
    reward, valid_count, invalid_count, supports_final = reasoning_consistency_reward(assistant_response)
    return supports_final

def get_num_valid_equations(assistant_response: str) -> int:
    """Returns the number of equations which are true (regardless of relevance to the problem)"""
    reward, valid_count, invalid_count, supports_final = reasoning_consistency_reward(assistant_response)
    return valid_count

def get_num_invalid_equations(assistant_response: str) -> int:
    """Returns the number of equations which are false (regardless of relevance to the problem)"""
    reward, valid_count, invalid_count, supports_final = reasoning_consistency_reward(assistant_response)
    return invalid_count

def get_num_valid_units(correct_response: str, assistant_response: str) -> int:
    """Returns the number of unit words used by the assistant which are relevant to the problem"""
    reward, num_valid_units, num_invalid_units, num_missing_units = calculate_units_reward(correct_response, assistant_response)
    return num_valid_units

def get_num_invalid_units(correct_response: str, assistant_response: str) -> int:
    """Returns the number of unit words used by the assistant which are irrelevant to the problem"""
    reward, num_valid_units, num_invalid_units, num_missing_units = calculate_units_reward(correct_response, assistant_response)
    return num_invalid_units

def get_num_missing_units(correct_response: str, assistant_response: str) -> int:
    """Returns the number of unit words relevant to the problem the assistant did not use"""
    reward, num_valid_units, num_invalid_units, num_missing_units = calculate_units_reward(correct_response, assistant_response)
    return num_missing_units

#--------------------------------------------------------------------------------
# Reward functions

def reasoning_consistency_reward(assistant_response: str) -> float:
    """
    Compute a reward for using consistent reasoning in the answer
    """
    reward = 0.0

    final_answer, confident = extract_final_answer(assistant_response)
    if final_answer is None:
        return 0.0

    if confident:
        reward += 0.05

    eqs = extract_equations(assistant_response)
    if not eqs:
        return reward

    valid_count = 0
    invalid_count = 0
    supports_final = False

    for a, op, b, c in eqs:
        calc = eval_equation(a, op, b)
        if calc is None:
            continue

        if abs(calc - c) < 1e-6:
            valid_count += 1
            if abs(c - final_answer) < 1e-6:
                supports_final = True
        else:
            invalid_count += 1

    if valid_count > 0:
        reward += 0.10
    if supports_final:
        reward += 0.15
    if invalid_count > 0:
        reward -= 0.10

    return reward, valid_count, invalid_count, supports_final

def calculate_units_reward(correct_response: str, assistant_response: str) -> float:
    """
    Compute a penalty (negative reward) for the use of incorrect units, or for missing units, in the model's answer,
    where units are e.g. "pound", "kilogram", "meter", "day".
    The penalty is the fraction of unit words that were used in the ground truth answer which
    the model failed to use. Also, if the model uses any incorrect units, the reward is -1.0, to
    discourage guessing / spraying unit words.
    If the ground truth answer uses no units, the penalty is 0.
    """
    assert isinstance(assistant_response, str), "Assuming simple string response for now"
    # Extract both the ground truth answer and the predicted answer
    ref_units = extract_units(correct_response)
    pred_units = extract_units(assistant_response)
    num_total_units = len(ref_units)
    if num_total_units == 0:
        return 0.0
    # Check if response used incorrect unit
    if len(set(pred_units) - set(ref_units)) > 0:
        return -1.0
    # Compute fraction of missing units
    num_missing_units = len(set(ref_units) - set(pred_units))
    return -num_missing_units / num_total_units, num_total_units - num_missing_units, len(set(pred_units) - set(ref_units)), num_missing_units

#--------------------------------------------------------------------------------
# Reasoning consistency helpers (by Steven)

def extract_final_answer(text):
    text_low = text.lower()
    patterns = [
        r"final answer[:\s]*\$?(-?\d+(?:\.\d+)?)",
        r"answer[:\s]*\$?(-?\d+(?:\.\d+)?)",
        r"the answer is[:\s]*\$?(-?\d+(?:\.\d+)?)",
        r"\\boxed\{(-?\d+(?:\.\d+)?)\}",
    ]

    for pat in patterns:
        m = re.search(pat, text_low)
        if m:
            return float(m.group(1)), True

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[-1]), False

    return None, False

def extract_equations(text: str):
    matches = re.findall(
        r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
        text
    )
    return [(float(a), op, float(b), float(c)) for a, op, b, c in matches]

def eval_equation(a: float, op: str, b: float):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if abs(b) < 1e-12:
            return None
        return a / b
    return None

#--------------------------------------------------------------------------------
# Units helpers (by Tyson)

def extract_units(response: str) -> set:
    """
    Return the set of unit strings used in the response,
    normalized to lowercase and singular for comparison.
    """
    units = set()
    # Remove numbers so we can find "mm" in "45mm" and "$" in "$20"
    response = re.sub(r'[0-9.,]', '', response)
    tokens = response.lower().split(' ')
    max_n = max(len(k.split(' ')) for k in UNITS)
    for n in range(max_n, 0, -1):
        n_grams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        for n_gram in n_grams:
            if n_gram in UNITS:
                units.add(UNITS[n_gram])
    return units

UNITS = {
    # Time
    "s": "second", "sec": "second", "secs": "second", "second": "second", "seconds": "second",
    "min": "minute", "mins": "minute", "minute": "minute", "minutes": "minute",
    "h": "hour", "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",
    "d": "day", "day": "day", "days": "day",
    "wk": "week", "wks": "week", "week": "week", "weeks": "week",
    "mo": "month", "mos": "month", "month": "month", "months": "month",
    "yr": "year", "yrs": "year", "year": "year", "years": "year",
    # Length
    "mm": "millimetre", "millimeter": "millimetre", "millimeters": "millimetre", "millimetre": "millimetre", "millimetres": "millimetre",
    "cm": "centimetre", "centimeter": "centimetre", "centimeters": "centimetre", "centimetre": "centimetre", "centimetres": "centimetre",
    "meter": "metre", "meters": "metre", "metre": "metre", "metres": "metre",
    "km": "kilometre", "kilometer": "kilometre", "kilometers": "kilometre", "kilometre": "kilometre", "kilometres": "kilometre",
    "in": "inch", "inch": "inch", "inches": "inch",
    "ft": "foot", "foot": "foot", "feet": "foot",
    "yd": "yard", "yard": "yard", "yards": "yard",
    "mi": "mile", "mile": "mile", "miles": "mile",
    # Mass
    "mg": "milligram", "milligram": "milligram", "milligrams": "milligram", "milligramme": "milligram", "milligrammes": "milligram",
    "g": "gram", "gram": "gram", "grams": "gram", "gramme": "gram", "grammes": "gram",
    "kg": "kilogram", "kilogram": "kilogram", "kilograms": "kilogram", "kilogramme": "kilogram", "kilogrammes": "kilogram",
    "oz": "ounce", "ounce": "ounce", "ounces": "ounce",
    "lb": "pound", "lbs": "pound", "pound": "pound", "pounds": "pound",
    "ton": "tonne", "tons": "tonne", "tonne": "tonne", "tonnes": "tonne",
    # Volume
    "ml": "millilitre", "milliliter": "millilitre", "milliliters": "millilitre", "millilitre": "millilitre", "millilitres": "millilitre",
    "l": "litre", "L": "litre", "liter": "litre", "liters": "litre", "litre": "litre", "litres": "litre",
    "cup": "cup", "cups": "cup",
    "pt": "pint", "pint": "pint", "pints": "pint",
    "qt": "quart", "quart": "quart", "quarts": "quart",
    "gal": "gallon", "gallon": "gallon", "gallons": "gallon",
    # Temperature
    # Note we parse all as "degree", including the angular degree, because
    # we can't reliably differentiate.
    "degree": "degree", "degrees": "degree", "celsius": "degree", "f": "degree", "°f": "degree", "fahrenheit": "degree",
    # Speed
    "m/s": "metre per second", "meter per second": "metre per second", "meters per second": "metre per second", "metre per second": "metre per second", "metres per second": "metre per second",
    "km/h": "kilometre per hour", "kph": "kilometre per hour", "kilometer per hour": "kilometre per hour", "kilometers per hour": "kilometre per hour", "kilometre per hour": "kilometre per hour", "kilometres per hour": "kilometre per hour",
    "mph": "mile per hour", "mile per hour": "mile per hour", "miles per hour": "mile per hour",
    # Money
    "$": "dollar", "dollar": "dollar", "dollars": "dollar",
    "cent": "cent", "cents": "cent",
}

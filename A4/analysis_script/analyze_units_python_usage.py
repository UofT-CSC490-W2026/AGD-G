#!/usr/bin/env python3
import argparse
import importlib.util
import json
from collections import Counter
import uniplot
import re

def count_python_calls(assistant_response: str) -> int:
    return len(re.findall(r'<\|python_start\|>', assistant_response))

def compute_unit_stats(correct_response: str, assistant_response: str) -> float:
    assert isinstance(assistant_response, str), "Assuming simple string response for now"
    # Extract both the ground truth answer and the predicted answer
    ref_units = extract_units(correct_response)
    pred_units = extract_units(assistant_response)
    num_total_units = len(ref_units)
    # Compute fraction of missing units
    num_missing_units = len(set(ref_units) - set(pred_units))
    return num_total_units - num_missing_units, len(set(pred_units) - set(ref_units)), num_missing_units

def extract_units(response: str) -> set:
    """
    Return the set of unit strings used in the response,
    normalized to lowercase and singular for comparison.
    """
    units = set()
    # Remove numbers so we can find "mm" in "45mm" and "$" in "$20"
    response = re.sub(r"[^a-zA-Z/$' ]", ' ', response)
    tokens = [token.strip() for token in response.lower().split(' ') if token.strip() != '']
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
    "inch": "inch", "inches": "inch",
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

def main():
    parser = argparse.ArgumentParser(description="Analyze GSM8K prediction JSONL from chat_eval.")
    parser.add_argument("--preds", required=True, help="Path to *.GSM8K.jsonl predictions file")
    args = parser.parse_args()

    total = 0
    correct = 0
    pass_unit_valid = 0
    pass_unit_invalid = 0
    pass_unit_missing = 0
    pass_unit_unneeded = 0
    fail_unit_valid = 0
    fail_unit_invalid = 0
    fail_unit_missing = 0
    fail_unit_unneeded = 0
    pass_python_calls = []
    fail_python_calls = []

    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += 1
            passed = bool(row.get("passed", False))
            if passed:
                correct += 1

            pred_answer = row.get("pred_answer")
            gold_answer = row.get("gold_answer")
            pred_response = row.get("pred_response") or row.get("completion") or ""
            gold_response = row.get("gold_response", "")

            try:
                vu, iu, mu = compute_unit_stats(gold_response, pred_response)
                if passed:
                    pass_unit_invalid += int(iu > 0)
                    pass_unit_missing += int(mu > 0 and iu == 0)
                    pass_unit_valid += int(vu > 0 and iu == 0 and mu == 0)
                    pass_unit_unneeded += int(vu == 0 and iu == 0 and mu == 0)
                else:
                    fail_unit_valid += int(vu > 0 and iu == 0 and mu == 0)
                    fail_unit_invalid += int(iu > 0)
                    fail_unit_missing += int(mu > 0 and iu == 0)
                    fail_unit_unneeded += int(vu == 0 and iu == 0 and mu == 0)
            except Exception:
                pass
            if passed:
                pass_python_calls.append(count_python_calls(pred_response))
            else:
                fail_python_calls.append(count_python_calls(pred_response))

    acc = (correct / total) if total else 0.0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")

    incorrect = total - correct
    total_vu = pass_unit_valid + fail_unit_valid
    total_iu = pass_unit_invalid + fail_unit_invalid
    total_mu = pass_unit_missing + fail_unit_valid

    print(f"#valid,pass     = {pass_unit_valid}")
    print(f"#invalid,pass   = {pass_unit_invalid}")
    print(f"#missing,pass   = {pass_unit_missing}")
    print(f"#unneeded,pass  = {pass_unit_unneeded}")
    print(f"#valid,fail     = {fail_unit_valid}")
    print(f"#invalid,fail   = {fail_unit_invalid}")
    print(f"#missing,fail   = {fail_unit_missing}")
    print(f"#unneeded,fail  = {fail_unit_unneeded}")
    print("Probabilities of using units and passing numerical correctness")
    print(f"P(valid|pass)   = {pass_unit_valid / correct:.3f}")
    print(f"P(invalid|pass) = {pass_unit_invalid / correct:.3f}")
    print(f"P(missing|pass) = {pass_unit_missing / correct:.3f}")
    print(f"P(unneeded|pass)= {pass_unit_unneeded / correct:.3f}")
    print(f"P(valid|fail)   = {fail_unit_valid / incorrect:.3f}")
    print(f"P(invalid|fail) = {fail_unit_invalid / incorrect:.3f}")
    print(f"P(missing|fail) = {fail_unit_missing / incorrect:.3f}")
    print(f"P(unneeded|fail)= {fail_unit_unneeded / incorrect:.3f}")
    print(f"P(pass|valid)   = {pass_unit_valid / total_vu:.3f}")
    print(f"P(pass|invalid) = {pass_unit_invalid / total_iu:.3f}")
    print(f"P(pass|missing) = {pass_unit_missing / total_mu:.3f}")
    print(f"P(pass|unneeded) = {pass_unit_unneeded / total_mu:.3f}")

    # Analyze number of python calls
    pass_used_python = sum(1 for n in pass_python_calls if n > 0)
    pass_no_used_python = sum(1 for n in pass_python_calls if n == 0)
    fail_used_python = sum(1 for n in fail_python_calls if n > 0)
    fail_no_used_python = sum(1 for n in fail_python_calls if n == 0)
    print()
    print(f"#python,pass       = {pass_used_python}")
    print(f"#no python,pass    = {pass_no_used_python}")
    print(f"#python,fail       = {fail_used_python}")
    print(f"#no python,fail    = {fail_no_used_python}")
    print("Probailities of using Python and correctness")
    print(f"P(python,pass)    = {pass_used_python / total:.3f}")
    print(f"P(python,fail)    = {fail_used_python / total:.3f}")
    print(f"P(no python,pass) = {pass_no_used_python / total:.3f}")
    print(f"P(no python,fail) = {fail_no_used_python / total:.3f}")
    print(f"P(python|pass)    = {pass_used_python / correct:.3f}")
    print(f"P(python|fail)    = {fail_used_python / incorrect:.3f}")
    print(f"P(pass|python)    = {pass_used_python / (pass_used_python + fail_used_python):.3f}")
    print(f"P(pass|no python) = {pass_no_used_python / (pass_no_used_python + fail_no_used_python):.3f}")
    max_python_calls = max(pass_python_calls + fail_python_calls)
    min_python_calls = min(pass_python_calls + fail_python_calls)
    uniplot.histogram(
            [pass_python_calls, fail_python_calls],
            legend_labels=["pass", "fail"],
            title='# Python calls for each response',
            color=['green', 'red'],
            bins=max_python_calls - min_python_calls + 1,
            bins_min=min_python_calls-0.5,
            bins_max=max_python_calls+0.5,
            character_set="braille",
        )

if __name__ == "__main__":
    main()

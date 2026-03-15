import argparse
import json
import re
from statistics import mean, median

# =========================================================
# Patterns
# =========================================================

PYTHON_PATTERN = re.compile(
    r"<\|python_start\|>.*?<\|python_end\|>|<\|output_start\|>.*?<\|output_end\|>",
    re.DOTALL,
)

OUTPUT_PATTERN = re.compile(
    r"<\|output_start\|>(.*?)<\|output_end\|>",
    re.DOTALL,
)

FINAL_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")
TRAILING_NUM_PATTERN = re.compile(r"([-+]?\d[\d,]*\.?\d*)\s*$")


# =========================================================
# Helpers
# =========================================================


def word_len(text):
    return len((text or "").split())


def char_len(text):
    return len(text or "")


def safe_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pct(n, d):
    return 0.0 if d == 0 else n / d


def summarize_numeric(values):
    if not values:
        print("no numeric values available")
        return
    print(f"count: {len(values)}")
    print(f"mean: {mean(values):.4f}")
    print(f"median: {median(values):.4f}")
    print(f"min: {min(values):.4f}")
    print(f"max: {max(values):.4f}")


def make_bin_label(lo, hi):
    if hi is None:
        return f"{lo}+"
    return f"{lo}-{hi}"


def bin_value(x, edges):
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if hi is None:
            if x >= lo:
                return make_bin_label(lo, hi)
        else:
            if lo <= x < hi:
                return make_bin_label(lo, hi)
    return "unknown"


def print_group_accuracy(rows, key_name, title, sort_fn=None):
    print(f"\n===== {title} =====")
    grouped = {}
    for r in rows:
        key = r[key_name]
        grouped.setdefault(key, []).append(r)

    keys = list(grouped.keys())
    if sort_fn is not None:
        keys = sorted(keys, key=sort_fn)
    else:
        try:
            keys = sorted(keys)
        except TypeError:
            pass

    for k in keys:
        grp = grouped[k]
        n = len(grp)
        c = sum(1 for r in grp if r["passed"])
        print(f"{k}: {c}/{n} ({pct(c, n):.3f})")


# =========================================================
# Extraction logic
# =========================================================


def clean_model_response(text):
    return PYTHON_PATTERN.sub("", text or "").strip()


def extract_expected_response(example):
    messages = example.get("conversation", {}).get("messages", [])

    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", [])
            texts = []

            if isinstance(content, str):
                return content.strip()

            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))

            return "\n".join(texts).strip()

    return ""


def extract_question(example):
    messages = example.get("conversation", {}).get("messages", [])
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "\n".join(parts).strip()
    return ""


def extract_output_numbers(text):
    nums = []
    for block in OUTPUT_PATTERN.findall(text or ""):
        nums += NUMBER_PATTERN.findall(block)

    out = []
    for x in nums:
        try:
            out.append(float(x.replace(",", "")))
        except ValueError:
            pass
    return out


def extract_final_answer(text):
    if not text:
        return None

    m = FINAL_PATTERN.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None

    cleaned = clean_model_response(text)
    m2 = TRAILING_NUM_PATTERN.search(cleaned)
    if m2:
        try:
            return float(m2.group(1).replace(",", ""))
        except ValueError:
            return None

    return None


def parse_gold_answer(example):
    ans = safe_float(example.get("gold_answer"))
    if ans is not None:
        return ans

    ans = extract_final_answer(example.get("gold_response", ""))
    if ans is not None:
        return ans

    return extract_final_answer(extract_expected_response(example))


def parse_pred_answer(example):
    ans = safe_float(example.get("pred_answer"))
    if ans is not None:
        return ans
    return extract_final_answer(example.get("pred_response", ""))


def count_python_outputs(text):
    return len(OUTPUT_PATTERN.findall(text or ""))


def type_token_ratio(text):
    tokens = re.findall(r"\b\w+\b", (text or "").lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# =========================================================
# Main EDA
# =========================================================


def run_eda(path):
    rows = []

    right_pred = []
    wrong_pred = []
    right_gt = []
    wrong_gt = []
    correct_consistency = []
    wrong_consistency = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)

            pred_raw = ex.get("pred_response", "")
            pred_clean = clean_model_response(pred_raw)
            gt = extract_expected_response(ex)
            question = extract_question(ex)
            passed = ex.get("passed", False)

            output_nums = extract_output_numbers(pred_raw)
            final_ans = extract_final_answer(pred_raw)

            consistent = False
            if final_ans is not None:
                for x in output_nums:
                    if abs(x - final_ans) < 1e-6:
                        consistent = True
                        break

            gold_num = parse_gold_answer(ex)
            pred_num = parse_pred_answer(ex)

            abs_error = None
            if gold_num is not None and pred_num is not None:
                abs_error = abs(pred_num - gold_num)

            row = {
                "passed": passed,
                "question": question,
                "question_words": word_len(question),
                "question_chars": char_len(question),
                "pred_raw": pred_raw,
                "pred_clean": pred_clean,
                "pred_words": word_len(pred_clean),
                "pred_chars": char_len(pred_clean),
                "pred_ttr": type_token_ratio(pred_clean),
                "gt": gt,
                "gt_words": word_len(gt),
                "gt_chars": char_len(gt),
                "num_python_outputs": count_python_outputs(pred_raw),
                "consistent": consistent,
                "gold_num": gold_num,
                "pred_num": pred_num,
                "abs_error": abs_error,
            }
            rows.append(row)

            if passed:
                right_pred.append(pred_clean)
                right_gt.append(gt)
                correct_consistency.append(consistent)
            else:
                wrong_pred.append(pred_clean)
                wrong_gt.append(gt)
                wrong_consistency.append(consistent)

    total = len(rows)
    num_correct = sum(1 for r in rows if r["passed"])
    num_wrong = total - num_correct

    # =====================================================
    # Dataset summary
    # =====================================================

    print("\n===== DATASET SUMMARY =====")
    print(f"total examples: {total}")
    print(f"correct: {num_correct}")
    print(f"wrong: {num_wrong}")
    if total > 0:
        print(f"accuracy: {num_correct / total:.3f}")

    # =====================================================
    # Response length
    # =====================================================

    print("\n===== MODEL RESPONSE LENGTH =====")
    if right_pred:
        print("\ncorrect predictions (model response)")
        print(f"count: {len(right_pred)}")
        print(f"avg words: {mean(word_len(t) for t in right_pred):.2f}")
        print(f"avg chars: {mean(char_len(t) for t in right_pred):.2f}")

    if wrong_pred:
        print("\nwrong predictions (model response)")
        print(f"count: {len(wrong_pred)}")
        print(f"avg words: {mean(word_len(t) for t in wrong_pred):.2f}")
        print(f"avg chars: {mean(char_len(t) for t in wrong_pred):.2f}")

    print("\n===== EXPECTED RESPONSE LENGTH =====")
    if right_gt:
        print("\ncorrect predictions (expected)")
        print(f"count: {len(right_gt)}")
        print(f"avg words: {mean(word_len(t) for t in right_gt):.2f}")
        print(f"avg chars: {mean(char_len(t) for t in right_gt):.2f}")

    if wrong_gt:
        print("\nwrong predictions (expected)")
        print(f"count: {len(wrong_gt)}")
        print(f"avg words: {mean(word_len(t) for t in wrong_gt):.2f}")
        print(f"avg chars: {mean(char_len(t) for t in wrong_gt):.2f}")

    print("\n===== LENGTH DIFFERENCE =====")
    if right_pred and right_gt:
        diffs = [word_len(p) - word_len(g) for p, g in zip(right_pred, right_gt)]
        print(f"avg word diff (correct): {mean(diffs):.2f}")

    if wrong_pred and wrong_gt:
        diffs = [word_len(p) - word_len(g) for p, g in zip(wrong_pred, wrong_gt)]
        print(f"avg word diff (wrong): {mean(diffs):.2f}")

    # =====================================================
    # Question length
    # =====================================================

    print("\n===== QUESTION LENGTH =====")
    correct_q = [r["question_words"] for r in rows if r["passed"]]
    wrong_q = [r["question_words"] for r in rows if not r["passed"]]

    if correct_q:
        print(f"correct avg question words: {mean(correct_q):.2f}")
        print(
            f"correct avg question chars: {mean(r['question_chars'] for r in rows if r['passed']):.2f}"
        )
    if wrong_q:
        print(f"wrong avg question words: {mean(wrong_q):.2f}")
        print(
            f"wrong avg question chars: {mean(r['question_chars'] for r in rows if not r['passed']):.2f}"
        )

    question_bins = [0, 20, 40, 60, 80, None]
    for r in rows:
        r["question_len_bin"] = bin_value(r["question_words"], question_bins)

    print_group_accuracy(
        rows,
        "question_len_bin",
        "QUESTION LENGTH BINS VS ACCURACY",
        sort_fn=lambda x: (
            999999 if x == "unknown" else int(x.split("-")[0].replace("+", ""))
        ),
    )

    # =====================================================
    # Model response length bins
    # =====================================================

    response_bins = [0, 30, 60, 90, 120, None]
    for r in rows:
        r["response_len_bin"] = bin_value(r["pred_words"], response_bins)

    print_group_accuracy(
        rows,
        "response_len_bin",
        "MODEL RESPONSE LENGTH BINS VS ACCURACY",
        sort_fn=lambda x: (
            999999 if x == "unknown" else int(x.split("-")[0].replace("+", ""))
        ),
    )

    # =====================================================
    # Consistency-related analysis
    # =====================================================

    print("\n===== PYTHON OUTPUT CONSISTENCY =====")
    if correct_consistency:
        c = sum(correct_consistency)
        print(
            f"correct examples where python output matches final answer: "
            f"{c}/{len(correct_consistency)} ({c/len(correct_consistency):.3f})"
        )

    if wrong_consistency:
        w = sum(wrong_consistency)
        print(
            f"wrong examples where python output matches final answer: "
            f"{w}/{len(wrong_consistency)} ({w/len(wrong_consistency):.3f})"
        )

    print_group_accuracy(
        rows,
        "num_python_outputs",
        "PYTHON OUTPUT COUNT VS ACCURACY",
        sort_fn=lambda x: x,
    )

    print("\n===== CONSISTENCY CONFUSION MATRIX =====")
    correct_consistent = sum(1 for r in rows if r["passed"] and r["consistent"])
    correct_inconsistent = sum(1 for r in rows if r["passed"] and not r["consistent"])
    wrong_consistent = sum(1 for r in rows if (not r["passed"]) and r["consistent"])
    wrong_inconsistent = sum(
        1 for r in rows if (not r["passed"]) and (not r["consistent"])
    )

    print("rows = actual correctness, cols = reasoning consistency")
    print(f"correct & consistent:   {correct_consistent}")
    print(f"correct & inconsistent: {correct_inconsistent}")
    print(f"wrong & consistent:     {wrong_consistent}")
    print(f"wrong & inconsistent:   {wrong_inconsistent}")

    consistent_total = correct_consistent + wrong_consistent
    inconsistent_total = correct_inconsistent + wrong_inconsistent

    if consistent_total > 0:
        print(
            f"p(correct | consistent): {correct_consistent}/{consistent_total} ({correct_consistent/consistent_total:.3f})"
        )
    if inconsistent_total > 0:
        print(
            f"p(correct | inconsistent): {correct_inconsistent}/{inconsistent_total} ({correct_inconsistent/inconsistent_total:.3f})"
        )

    # =====================================================
    # Numerical error summary
    # =====================================================

    print("\n===== ERROR MAGNITUDE ANALYSIS =====")
    wrong_numeric_errors = [
        r["abs_error"]
        for r in rows
        if (not r["passed"]) and (r["abs_error"] is not None)
    ]
    summarize_numeric(wrong_numeric_errors)

    # =====================================================
    # Word diversity
    # =====================================================

    print("\n===== WORD DIVERSITY (TYPE-TOKEN RATIO) =====")
    correct_ttr = [r["pred_ttr"] for r in rows if r["passed"]]
    wrong_ttr = [r["pred_ttr"] for r in rows if not r["passed"]]

    if correct_ttr:
        print(f"correct avg TTR: {mean(correct_ttr):.4f}")
    if wrong_ttr:
        print(f"wrong avg TTR: {mean(wrong_ttr):.4f}")

    for r in rows:
        r["ttr_bin"] = (
            "0-0.4"
            if r["pred_ttr"] < 0.4
            else (
                "0.4-0.6"
                if r["pred_ttr"] < 0.6
                else "0.6-0.8" if r["pred_ttr"] < 0.8 else "0.8+"
            )
        )

    print_group_accuracy(
        rows,
        "ttr_bin",
        "WORD DIVERSITY BINS VS ACCURACY",
        sort_fn=lambda x: {"0-0.4": 0, "0.4-0.6": 1, "0.6-0.8": 2, "0.8+": 3}.get(
            x, 99
        ),
    )

    # =====================================================
    # Diagnostics
    # =====================================================

    print("\n===== DIAGNOSTIC SUMMARY =====")
    no_python = sum(1 for r in rows if r["num_python_outputs"] == 0)
    missing_final = sum(1 for r in rows if r["pred_num"] is None)
    numeric_wrong = sum(
        1 for r in rows if (not r["passed"]) and r["abs_error"] is not None
    )

    print(
        f"examples with no python outputs: {no_python}/{total} ({pct(no_python, total):.3f})"
    )
    print(
        f"examples with no final numeric answer: {missing_final}/{total} ({pct(missing_final, total):.3f})"
    )
    print(
        f"wrong examples with numeric error measurable: {numeric_wrong}/{num_wrong} ({pct(numeric_wrong, num_wrong):.3f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="Path to JSONL predictions file")
    args = parser.parse_args()
    run_eda(args.preds)

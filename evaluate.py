"""
evaluate.py - measures how well the Tamil Sentiment model does on a small paired test set.

WHAT IT DOES
------------
test_set.csv has 36 short movie-comment sentences (12 Positive, 12 Negative,
12 Mixed Feelings). Each one is written twice: in Tamil script and in typed Tanglish.

The script runs the SAME sentences through the model in these ways:

  A. Typed Tanglish          - the sentence as people type it (the best case)
  B. Tamil script, NEW       - Tamil script converted with the new converter in app.py
  C. Tamil script, OLD       - Tamil script converted with the old ITRANS pipeline
                               (skipped automatically if indic_transliteration is missing)
  D. Tamil script, RAW       - Tamil script sent to the model with no conversion at all

and prints accuracy, macro F1, per-class recall and a confusion matrix for each.
It also saves every prediction to eval_results.csv so you can inspect single sentences.

HOW TO RUN (in the folder that has app.py, model/ and test_set.csv)
    py evaluate.py

It reuses the model and converter from app.py, so there is one source of truth.
"""

import csv
import sys

import torch

# Re-use the model, tokenizer, labels and converter from app.py (importing it does not start the server).
from app import tokenizer, model, id2label, to_tanglish

TEST_FILE = "test_set.csv"
OUT_FILE = "eval_results.csv"
CLASSES = ["Negative", "Mixed Feelings", "Positive"]  # same order as id2label 0, 1, 2

# ---------------------------------------------------------------- old pipeline (for comparison only)
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    HAVE_OLD = True
except Exception:
    HAVE_OLD = False

_OLD_LEFTOVERS = {"ன": "n", "ண": "n", "ழ": "zh", "ள": "l", "ற": "r", "்": ""}


def old_convert(text):
    out = transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
    for tamil_char, latin in _OLD_LEFTOVERS.items():
        out = out.replace(tamil_char, latin)
    return out


# ---------------------------------------------------------------- helpers
def load_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        # the Tamil column may be called "tamil" or "tamil_script"
        r["tamil"] = r.get("tamil") or r.get("tamil_script") or ""
        if not r["tamil"] or not r.get("tanglish"):
            sys.exit(f"Row {r.get('id')}: needs both a Tamil-script sentence and a Tanglish sentence")
        if r["label"] not in CLASSES:
            sys.exit(f"Row {r['id']}: label '{r['label']}' must be one of {CLASSES}")
    return rows


def predict(texts, batch_size=16):
    """Returns a list of (label, confidence_percent) for each text."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(**enc).logits, dim=-1)
        for p in probs:
            idx = int(torch.argmax(p))
            results.append((id2label[idx], round(float(p[idx]) * 100, 1)))
    return results


def metrics(true, pred):
    n = len(true)
    acc = sum(t == p for t, p in zip(true, pred)) / n
    per_class, f1s = {}, []
    for c in CLASSES:
        tp = sum(t == c and p == c for t, p in zip(true, pred))
        fp = sum(t != c and p == c for t, p in zip(true, pred))
        fn = sum(t == c and p != c for t, p in zip(true, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = rec
        f1s.append(f1)
    conf = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for t, p in zip(true, pred):
        conf[t][p] += 1
    return acc, sum(f1s) / len(f1s), per_class, conf


def report(name, true, pred):
    acc, macro_f1, recall, conf = metrics(true, pred)
    print(f"\n=== {name} ===")
    print(f"accuracy {acc * 100:.1f}%   macro F1 {macro_f1:.2f}   ({sum(t == p for t, p in zip(true, pred))}/{len(true)} correct)")
    print("recall per class: " + "   ".join(f"{c} {recall[c] * 100:.0f}%" for c in CLASSES))
    print("confusion matrix (rows = true, columns = predicted):")
    print(f"{'':16}" + "".join(f"{c:>16}" for c in CLASSES))
    for t in CLASSES:
        print(f"{t:16}" + "".join(f"{conf[t][p]:>16}" for p in CLASSES))
    return acc, macro_f1


# ---------------------------------------------------------------- main
def main():
    rows = load_rows(TEST_FILE)
    true = [r["label"] for r in rows]

    conditions = {
        "A. Typed Tanglish": [r["tanglish"] for r in rows],
        "B. Tamil script -> NEW converter": [to_tanglish(r["tamil"]) for r in rows],
    }
    if HAVE_OLD:
        conditions["C. Tamil script -> OLD ITRANS pipeline"] = [old_convert(r["tamil"]) for r in rows]
    else:
        print("(indic_transliteration not installed: skipping the OLD pipeline comparison)")
    conditions["D. Tamil script -> RAW (no conversion)"] = [r["tamil"] for r in rows]

    preds, summary = {}, []
    for name, texts in conditions.items():
        preds[name] = predict(texts)
        acc, f1 = report(name, true, [p[0] for p in preds[name]])
        summary.append((name, acc, f1))

    print("\n\n=== SUMMARY ===")
    print(f"{'condition':44}{'accuracy':>10}{'macro F1':>10}")
    for name, acc, f1 in summary:
        print(f"{name:44}{acc * 100:>9.1f}%{f1:>10.2f}")
    print(f"\nTest set size: {len(rows)} sentences ({len(rows) // 3} per class). "
          "Small set: treat differences of 1-2 sentences as noise.")

    with open(OUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        header = ["id", "true_label", "tamil", "tanglish"]
        for name in conditions:
            tag = name.split(".")[0]
            header += [f"{tag}_input", f"{tag}_pred", f"{tag}_conf"]
        w.writerow(header)
        for i, r in enumerate(rows):
            line = [r["id"], r["label"], r["tamil"], r["tanglish"]]
            for name, texts in conditions.items():
                label, conf = preds[name][i]
                line += [texts[i], label, conf]
            w.writerow(line)
    print(f"\nPer-sentence results saved to {OUT_FILE} (open it in Excel to see every prediction).")


if __name__ == "__main__":
    main()

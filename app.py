"""
Tamil Sentiment Analysis — Flask Demo App (v5: new Tanglish converter)
======================================================================
Loads your fine-tuned Tamil-BERT model and serves a polished web page
where you can type a Tamil/Tanglish comment and see the predicted sentiment.

CHANGED IN v5:
The old ITRANS transliteration turned Tamil script into strings like
"bhadham" / "jhubhbhar" / "mòghgha", which look nothing like the Tanglish the
model was trained on (people type "padam" / "super" / "mokka"). The model saw
noise and fell back to its majority class (Positive). Typing the same
sentences in natural Tanglish gave correct answers.

This version replaces ITRANS with a built-in converter that writes Tamil the
way people actually type it in Latin letters (plain k / s / t / th / p, no
diacritics, English loanwords like super / waste / time spelled normally).
It is all inside this one file, and the indic_transliteration library is no
longer needed.

FOLDER STRUCTURE EXPECTED:
tamil_sentiment_app/
    app.py              <- this file
    model/
        config.json
        tokenizer_config.json
        training_args.bin
        model.safetensors
        tokenizer.json

RUN WITH:
    py app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import re
import unicodedata
from flask import Flask, render_template_string, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# --- Load model once, at startup ---
MODEL_PATH = "./model"

print("Loading model... this may take a moment.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = {0: "Negative", 1: "Mixed Feelings", 2: "Positive"}

print("Model loaded successfully.")


# =====================================================================
# Tamil script -> casual Tanglish converter
# =====================================================================
PULLI = "்"  # virama: removes the inherent "a" of a consonant
TAMIL_RUN = re.compile(r"[஀-௿]+")
TAMIL_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")

# --- letter tables ---
VOWELS = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo",
    "எ": "e", "ஏ": "e", "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au",
}
SIGNS = {  # dependent vowel signs
    "ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo",
    "ெ": "e", "ே": "e", "ை": "ai",
    "ொ": "o", "ோ": "o", "ௌ": "au",
}
CONSONANTS = {
    "க": "k", "ங": "ng", "ச": "s", "ஞ": "nj", "ட": "t", "ண": "n", "த": "th",
    "ந": "n", "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v",
    "ழ": "zh", "ள": "l", "ற": "r", "ன": "n",
    "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h", "ஶ": "sh", "ஃ": "h",
}
STOPS = set("கசடதபற")

# How a stop consonant is typed depending on where it sits in the word.
INITIAL = {"க": "k", "ச": "s", "ட": "t", "த": "th", "ப": "p", "ற": "r"}
INTERVOCALIC = {"க": "g", "ச": "s", "ட": "d", "த": "th", "ப": "p", "ற": "r"}
AFTER_NASAL = {"க": "g", "ச": "j", "ட": "d", "த": "th", "ப": "b", "ற": "dr"}
GEMINATE = {"க": "kk", "ச": "ch", "ட": "tt", "த": "th", "ப": "pp", "ற": "tr"}
# nasal -> the stop it normally combines with (ngk, nj-ch, nd, nth, mp, ndr)
NASAL_STOP = {"ங": "க", "ஞ": "ச", "ண": "ட", "ந": "த", "ம": "ப", "ன": "ற"}

# --- word dictionary ---
# English loanwords and a few very common words whose spelling rules alone get wrong.
# Keys that end in a pulli also match with a vowel suffix: சூப்பரா -> supera.
# If a word still comes out odd, add it here.
_WORDS = {
    # reactions / quality
    "சூப்பர்": "super", "சூப்பர்ஹிட்": "superhit", "மாஸ்": "mass", "கிளாஸ்": "class",
    "லெவல்": "level", "ஹிட்": "hit", "பிளாப்": "flop", "ப்ளாப்": "flop",
    "பிளாக்பஸ்டர்": "blockbuster", "டாப்": "top", "பெஸ்ட்": "best",
    "வொர்ஸ்ட்": "worst", "குட்": "good", "பேட்": "bad", "நைஸ்": "nice",
    "கூல்": "cool", "ஹாப்பி": "happy", "ஓகே": "ok", "லவ்": "love",
    "ஸ்டைல்": "style", "எக்ஸலன்ட்": "excellent", "எக்ஸலண்ட்": "excellent",
    "ஒர்த்": "worth", "ஒரிஜினல்": "original", "ஃபீல்": "feel", "பீல்": "feel",
    # negative
    "வேஸ்ட்": "waste", "வேஸ்ட்டு": "waste", "வேஸ்ட்டா": "waste",
    "போர்": "bore", "போரிங்": "boring",
    # movie vocabulary
    "ஹீரோ": "hero", "ஹீரோயின்": "heroine", "வில்லன்": "villain",
    "டைரக்டர்": "director", "டைரக்ஷன்": "direction", "மியூசிக்": "music",
    "பிஜிஎம்": "bgm", "கிளைமாக்ஸ்": "climax", "கிளைமேக்ஸ்": "climax",
    "இன்டர்வல்": "interval", "ஸ்டோரி": "story", "ஸ்கிரீன்பிளே": "screenplay",
    "ஸ்க்ரீன்பிளே": "screenplay", "ட்ரெய்லர்": "trailer", "டிரெய்லர்": "trailer",
    "ட்ரைலர்": "trailer", "டீசர்": "teaser", "தியேட்டர்": "theatre",
    "ஓடிடி": "ott", "ஃபர்ஸ்ட்": "first", "பர்ஸ்ட்": "first", "ஹாஃப்": "half",
    "செகண்ட்": "second", "ஆக்ஷன்": "action", "ஆக்டிங்": "acting",
    "காமெடி": "comedy", "ரொமான்ஸ்": "romance", "எமோஷனல்": "emotional",
    "ஃபைட்": "fight", "டான்ஸ்": "dance", "சாங்": "song", "டயலாக்": "dialogue",
    "ரிவ்யூ": "review", "ரேட்டிங்": "rating", "டிக்கெட்": "ticket",
    "ரீமேக்": "remake", "சினிமா": "cinema", "மூவி": "movie", "ஃபிலிம்": "film",
    "வீடியோ": "video", "சேனல்": "channel", "லைக்": "like", "ஷேர்": "share",
    "கமெண்ட்": "comment", "சப்ஸ்கிரைப்": "subscribe", "ஃபேன்": "fan",
    "டைம்": "time", "ப்ரோ": "bro", "தேங்க்ஸ்": "thanks", "சாரி": "sorry",
    "ஃப்ரீ": "free",
    # very common colloquial words
    "டா": "da", "டி": "di", "டேய்": "dei", "மச்சான்": "machan",
}

WORDS = {unicodedata.normalize("NFC", k): v for k, v in _WORDS.items()}
# loanwords ending in a pulli can take vowel suffixes (சூப்பரா, சூப்பருங்க ...)
LOAN_STEMS = sorted(
    ((k[:-1], v) for k, v in WORDS.items() if k.endswith(PULLI)),
    key=lambda kv: -len(kv[0]),
)
SUFFIX_SIGNS = set("ாுேைி")  # aa u e ai i


def _parse(word):
    """Split a run of Tamil characters into (consonant, vowel, dead) units."""
    units, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        if ch in CONSONANTS:
            nxt = word[i + 1] if i + 1 < n else ""
            if ch == "ஃ":
                units.append((ch, None, True))
                i += 1
            elif nxt == PULLI:
                units.append((ch, None, True))
                i += 2
            elif nxt in SIGNS:
                units.append((ch, nxt, False))
                i += 2
            else:
                units.append((ch, None, False))  # inherent "a"
                i += 1
        elif ch in VOWELS or ch in SIGNS:  # independent vowel, or a stray sign (suffix)
            units.append((None, ch, False))
            i += 1
        else:  # anything else in the Tamil block is ignored
            i += 1
    return units


def _vowel(v, is_final):
    if v is None:
        return "a"
    if v in SIGNS:
        # a word-final long aa is typed as a single "a": nalla, kandippa, romba
        return "a" if (v == "ா" and is_final) else SIGNS[v]
    return VOWELS[v]


def _stop(c, prev):
    if prev is not None:
        pc, _, pdead = prev
        if pdead and NASAL_STOP.get(pc) == c:
            return AFTER_NASAL[c]
        if not pdead:  # previous unit ends in a vowel -> soft, between vowels
            return INTERVOCALIC[c]
    return INITIAL[c]


def _convert_word(word):
    units = _parse(word)
    out, skip, n = [], False, len(units)
    for i, (c, v, dead) in enumerate(units):
        final = i == n - 1
        if c is None:  # independent vowel or stray vowel sign
            out.append(_vowel(v, final))
            continue
        nxt = units[i + 1] if i + 1 < n else None
        if dead:
            if skip:
                skip = False
                continue
            if c in STOPS and nxt and nxt[0] == c:  # doubled stop: type it once
                out.append(GEMINATE[c])
                skip = True
                continue
            lat = CONSONANTS[c]
            if nxt and nxt[0] is not None and NASAL_STOP.get(c) == nxt[0]:
                if c in "ஙஞ":  # the following stop supplies the "g" / "j"
                    lat = "n"
            out.append(lat)
            continue
        prev = units[i - 1] if i > 0 else None
        if skip:
            lat, skip = "", False
        elif c in STOPS:
            lat = _stop(c, prev)
        else:
            lat = CONSONANTS[c]
        out.append(lat + _vowel(v, final))
    return "".join(out)


def _convert_token(tok):
    if tok in WORDS:
        return WORDS[tok]
    for stem, latin in LOAN_STEMS:
        if tok.startswith(stem) and len(tok) > len(stem) and tok[len(stem)] in SUFFIX_SIGNS:
            base = latin[:-1] if latin.endswith("e") else latin
            return base + _convert_word(tok[len(stem):])
    return _convert_word(tok)


def contains_tamil_script(text: str) -> bool:
    """Checks if the text contains any Tamil Unicode characters."""
    return bool(TAMIL_RUN.search(text or ""))


def to_tanglish(text: str) -> str:
    """Converts Tamil script to casual Tanglish. Non-Tamil text is left untouched."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "")
    text = text.translate(TAMIL_DIGITS)
    out = TAMIL_RUN.sub(lambda m: _convert_token(m.group()), text)
    return re.sub(r"\s+", " ", out).strip()


# =====================================================================
# Prediction
# =====================================================================
def preprocess_text(text: str):
    """
    If Tamil script is detected: convert it to casual Tanglish with to_tanglish()
    so the model sees the same kind of text it was trained on. That converted
    text is both what the model analyzes and what the UI note displays.

    Returns (model_input_text, display_text, was_converted).
    """
    if contains_tamil_script(text):
        converted = to_tanglish(text)
        return converted, converted, True
    return text, text, False


def predict_sentiment(text: str):
    model_input_text, display_text, was_transliterated = preprocess_text(text)

    inputs = tokenizer(model_input_text, return_tensors="pt", truncation=True,
                        padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    confidence = float(probs[pred_id])
    all_scores = {id2label[i]: round(float(probs[i]) * 100, 1) for i in range(3)}

    return id2label[pred_id], round(confidence * 100, 1), all_scores, display_text, was_transliterated


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Tamil Sentiment Analyzer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
            max-width: 680px;
            margin: 0 auto;
            padding: 48px 24px 80px;
            color: #1e293b;
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 36px;
        }
        .header .badge {
            display: inline-block;
            background: #eef2ff;
            color: #4338ca;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 4px 12px;
            border-radius: 999px;
            margin-bottom: 14px;
        }
        h1 {
            font-size: 32px;
            font-weight: 700;
            margin: 0 0 8px;
            background: linear-gradient(90deg, #4338ca, #7c3aed);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        p.subtitle {
            color: #64748b;
            margin: 0;
            font-size: 15px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 28px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.04);
            border: 1px solid #eef1f5;
        }
        textarea {
            width: 100%;
            height: 110px;
            font-size: 16px;
            font-family: inherit;
            padding: 14px 16px;
            border: 1.5px solid #e2e8f0;
            border-radius: 10px;
            resize: vertical;
            transition: border-color 0.15s ease;
        }
        textarea:focus {
            outline: none;
            border-color: #6366f1;
        }
        button {
            margin-top: 14px;
            width: 100%;
            padding: 13px 24px;
            font-size: 15px;
            font-weight: 600;
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: opacity 0.15s ease, transform 0.1s ease;
        }
        button:hover { opacity: 0.92; }
        button:active { transform: scale(0.99); }
        .result {
            margin-top: 24px;
            padding-top: 24px;
            border-top: 1px solid #eef1f5;
        }
        .label-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        .label-icon {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }
        .label-text {
            font-size: 20px;
            font-weight: 700;
        }
        .confidence-text {
            font-size: 13px;
            color: #94a3b8;
            font-weight: 500;
        }
        .positive .label-icon { background: #dcfce7; color: #16a34a; }
        .positive .label-text { color: #16a34a; }
        .negative .label-icon { background: #fee2e2; color: #dc2626; }
        .negative .label-text { color: #dc2626; }
        .mixed .label-icon { background: #fef3c7; color: #d97706; }
        .mixed .label-text { color: #d97706; }

        .bar-row { display: flex; align-items: center; margin-bottom: 10px; gap: 12px; }
        .bar-label { width: 110px; font-size: 13px; color: #64748b; font-weight: 500; }
        .bar-track { flex: 1; background: #f1f5f9; border-radius: 6px; height: 10px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #6366f1, #8b5cf6); }
        .bar-pct { width: 44px; text-align: right; font-size: 12px; color: #94a3b8; font-weight: 600; }

        .transliteration-note {
            margin-top: 18px;
            padding: 12px 16px;
            background: #f5f3ff;
            border: 1px solid #ddd6fe;
            border-radius: 10px;
            font-size: 13px;
            color: #5b21b6;
            line-height: 1.5;
        }
        .transliteration-note strong {
            font-family: 'Courier New', monospace;
            background: white;
            padding: 1px 6px;
            border-radius: 4px;
        }
        .note {
            margin-top: 28px;
            font-size: 12.5px;
            color: #94a3b8;
            text-align: center;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="header">
        <span class="badge">Tamil-BERT &middot; Fine-tuned</span>
        <h1>Tamil Sentiment Analyzer</h1>
        <p class="subtitle">Code-mixed (Tanglish) sentiment classification with automatic script handling</p>
    </div>

    <div class="card">
        <form method="POST">
            <textarea name="text" placeholder="Type a Tamil or Tanglish comment here...">{{ input_text }}</textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>

        {% if label %}
        <div class="result {{ label_class }}">
            <div class="label-row">
                <div class="label-icon">
                    {% if label_class == 'positive' %}😊{% elif label_class == 'negative' %}😞{% else %}😐{% endif %}
                </div>
                <div>
                    <div class="label-text">{{ label }}</div>
                    <div class="confidence-text">{{ confidence }}% confidence</div>
                </div>
            </div>

            {% for name, score in scores.items() %}
            <div class="bar-row">
                <div class="bar-label">{{ name }}</div>
                <div class="bar-track"><div class="bar-fill" style="width: {{ score }}%;"></div></div>
                <div class="bar-pct">{{ score }}%</div>
            </div>
            {% endfor %}

            {% if was_transliterated %}
            <div class="transliteration-note">
                🔤 Tamil script detected — converted to <strong>{{ processed_text }}</strong> before analysis.
            </div>
            {% endif %}
        </div>
        {% endif %}
    </div>

    <p class="note">
        Trained on the DravidianCodeMix dataset (Chakravarthi et al.) &middot;
        Best on clear positive/negative sentiment and casual/colloquial phrasing.<br>
        Formal or literary vocabulary may be misclassified — see project README for details.
    </p>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    label = None
    confidence = None
    scores = None
    input_text = ""
    processed_text = None
    was_transliterated = False

    if request.method == "POST":
        input_text = request.form.get("text", "").strip()
        if input_text:
            label, confidence, scores, processed_text, was_transliterated = predict_sentiment(input_text)

    label_class = ""
    if label == "Positive":
        label_class = "positive"
    elif label == "Negative":
        label_class = "negative"
    elif label == "Mixed Feelings":
        label_class = "mixed"

    return render_template_string(
        PAGE_TEMPLATE,
        label=label,
        label_class=label_class,
        confidence=confidence,
        scores=scores,
        input_text=input_text,
        processed_text=processed_text,
        was_transliterated=was_transliterated,
    )


if __name__ == "__main__":
    app.run(debug=True)

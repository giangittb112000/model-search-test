from pathlib import Path

import spacy
from spacy.tokens import DocBin

from data.dev_data import DEV_DATA
from data.train_data import TRAIN_DATA

OUTPUT_DIR = Path("data")
TRAIN_PATH = OUTPUT_DIR / "train.spacy"
DEV_PATH = OUTPUT_DIR / "dev.spacy"


def verify_offsets(examples, name):
    errors = []
    for text, annot in examples:
        for start, end, label in annot["entities"]:
            got = text[start:end]
            if not got or got.strip() == "":
                errors.append(f"[{name}] offset sai ({start},{end},{label}) in {text!r}")
    return errors


def build_docbin(nlp, examples):
    db = DocBin()
    skipped = 0
    for text, annot in examples:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is None:
                print(f"[WARN] bỏ qua entity ({start},{end},{label}) — không khớp token: {text!r}")
                skipped += 1
                continue
            ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db, skipped


def main():
    errors = verify_offsets(TRAIN_DATA, "train") + verify_offsets(DEV_DATA, "dev")
    if errors:
        for e in errors:
            print(e)
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp = spacy.blank(
        "vi",
        config={
            "nlp": {
                "tokenizer": {
                    "@tokenizers": "spacy.vi.VietnameseTokenizer",
                    "use_pyvi": False,
                }
            }
        },
    )

    train_db, train_skipped = build_docbin(nlp, TRAIN_DATA)
    dev_db, dev_skipped = build_docbin(nlp, DEV_DATA)

    train_db.to_disk(TRAIN_PATH)
    dev_db.to_disk(DEV_PATH)

    total = len(TRAIN_DATA) + len(DEV_DATA)
    print(
        f"Train: {len(TRAIN_DATA)} câu ({train_skipped} entity bỏ) → {TRAIN_PATH}\n"
        f"Dev:   {len(DEV_DATA)} câu ({dev_skipped} entity bỏ) → {DEV_PATH}\n"
        f"Tỷ lệ: {len(TRAIN_DATA)}/{total} train, {len(DEV_DATA)}/{total} dev"
    )


if __name__ == "__main__":
    main()

import spacy

from data.test_data import TEST_QUERIES
from device_env import apply_run_device, get_run_settings, model_best_path


def main():
    settings = get_run_settings()
    model_dir = model_best_path()

    if not model_dir.exists():
        print(f"Chưa có model tại {model_dir}")
        print("Chạy: make preprocess && make train")
        raise SystemExit(2)

    gpu_error = apply_run_device(spacy, settings)
    if gpu_error:
        print(f"[WARN] NER_RUN_MODE=gpu → fallback CPU: {gpu_error}")

    nlp = spacy.load(model_dir)
    print(f"Model: {model_dir}")
    print(f"NER_RUN_MODE: {settings.mode.upper()}  |  NER_GPU_ID: {settings.gpu_id}\n")

    for query in TEST_QUERIES:
        doc = nlp(query)
        print(f"Câu: {query}")
        if not doc.ents:
            print("  (không tìm thấy entity)\n")
            continue
        for ent in doc.ents:
            print(f"  {ent.label_:<10} | {ent.text!r}")
        print()


if __name__ == "__main__":
    main()

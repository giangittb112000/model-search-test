import sys
import time

import spacy

from device_env import apply_run_device, get_run_settings, model_best_path


def main():
    if len(sys.argv) < 2:
        print('Cách dùng: python src/predict.py "câu cần test"')
        sys.exit(1)

    settings = get_run_settings()
    model_dir = model_best_path()

    if not model_dir.exists():
        print(f"Chưa có model tại {model_dir}")
        print(f"(NER_TRAIN_MODE={settings.model_mode} → folder models/{settings.model_mode}/)")
        print("Chạy: make preprocess && make train")
        sys.exit(2)

    gpu_error = apply_run_device(spacy, settings)
    if gpu_error:
        print(f"[WARN] NER_RUN_MODE=gpu → fallback CPU: {gpu_error}")

    text = " ".join(sys.argv[1:])

    t0 = time.perf_counter()
    nlp = spacy.load(model_dir)
    t_load = time.perf_counter()
    doc = nlp(text)
    t_infer = time.perf_counter()

    print(f"Model:   {model_dir}")
    print(f"NER_RUN_MODE: {settings.mode.upper()}  |  NER_GPU_ID: {settings.gpu_id}")
    print(f"Câu:     {doc.text}")
    if not doc.ents:
        print("  (không tìm thấy entity)")
    else:
        for ent in doc.ents:
            print(f"  {ent.label_:<10} | {ent.text!r}")
    print(
        f"Thời gian: load {(t_load - t0) * 1000:.1f}ms | "
        f"infer {(t_infer - t_load) * 1000:.1f}ms"
    )


if __name__ == "__main__":
    main()

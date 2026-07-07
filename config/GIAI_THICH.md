# Giải thích config train

File `config.cfg` là cài đặt cho `make train`. MVP thường **không cần sửa**.

CPU/GPU → [docs/env.md](../docs/env.md) (3 biến).

---

## 1. Config làm gì?

Bảo spaCy: học từ file nào, chấm điểm bằng file nào, học bao lâu, lưu model thế nào.

---

## 2. Setting quan trọng

| Setting | Giá trị | Ý nghĩa |
| ------- | ------- | ------- |
| `eval_frequency` | `200` | Cứ 200 step thì thi thử trên dev |
| `max_steps` | `20000` | Học tối đa 20.000 step |
| `learn_rate` | `0.001` | Tốc độ học |
| `lang` | `"vi"` | Tiếng Việt |
| `use_pyvi` | `false` | Tokenize theo khoảng trắng |
| `gpu_allocator` | `null` | Chia bộ nhớ GPU lúc train — ghi đè bằng `NER_GPU_ALLOCATOR` trong `.env` |

---

## 3. File data

Khi `make train`, đường dẫn được truyền qua CLI:

```
train → data/train.spacy   (~150 câu) — model học
dev   → data/dev.spacy     (~36 câu)  — chấm điểm, chọn model-best
```

Test (`data/test_data.py`) không vào train — dùng `make test-predict`.

Tỷ lệ ~80% train / 20% dev. Dev không trùng train (auto-check).

---

## 4. Pipeline

```
"mua iphone 15 giá rẻ"
        ↓
   [Tách từ]     whitespace: mua | iphone | 15 | giá | rẻ
        ↓
   [tok2vec]     Vector hóa từng token
        ↓
   [ner]         Gắn nhãn CATEGORY / PRODUCT / SPEC
        ↓
   doc.ents
```

`pipeline = ["tok2vec", "ner"]` — hai bước trên.

---

## 5. Step và F-score

1. **Step** = một lần cập nhật weights từ train data.
2. Cứ `eval_frequency` step → thi trên dev → **F-score**.
3. Điểm cao nhất → `model-best/`. Step cuối → `model-last/` (chỉ debug).

---

## 6. Từ khó

| Từ | Nghĩa |
| --- | ----- |
| **weights** | Tham số model — cập nhật mỗi step |
| **loss** | Mức độ sai — càng thấp càng tốt |
| **dropout** | Bỏ ngẫu nhiên một phần khi học — tránh học vẹt |
| **overfit** | Thuộc train nhưng kém trên câu mới |

---

## 7. Phần config còn lại

| Phần | Cần sửa MVP? |
| ---- | ------------ |
| `[paths]`, `[training]` | Không — xem mục 2–3 |
| `[components.*]` | Không — kiến trúc mặc định spaCy |
| `[training.optimizer]` | Không — Adam default |

Tune sau: [spaCy training docs](https://spacy.io/usage/training).

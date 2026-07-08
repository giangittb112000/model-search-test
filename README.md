# spaCy NER (Vietnamese) - Docker MVP

Nhận diện **CATEGORY**, **PRODUCT**, **SPEC** trong câu tìm kiếm e-commerce.
Chạy trong Docker — không cần cài Python trên máy.

**PRICE** và **STOPWORD** xử lý bằng logic (regex + word list), không thuộc model — xem [docs/training_data_spec.md](docs/training_data_spec.md) §2.3.

---

## Chạy nhanh

**Yêu cầu:** Docker Desktop (hoặc Docker Engine + Compose), `make`.

```bash
make env-init-cpu    # bắt buộc lần đầu — tạo .env từ mẫu
make build
make preprocess
make train
make predict Q="laptop dell 16gb"
make test-predict
```

Source/config/data được copy vào Docker image khi `make build`. Không bind mount project vào container.
Sau train, model nằm trong filesystem của container đang chạy tại `models/cpu/` hoặc `models/gpu/` (theo `NER_TRAIN_MODE` trong `.env`).

Sửa data rồi chạy lại:

```bash
make preprocess && make train && make test-predict
```

---

## Biến môi trường

Mọi lệnh dùng **`.env`** — 4 biến: `NER_TRAIN_MODE`, `NER_RUN_MODE`, `NER_GPU_ID`, `NER_GPU_ALLOCATOR`.

```bash
make env-init-cpu    # hoặc make env-init-gpu
```

Chi tiết: **[docs/env.md](docs/env.md)**.

---

## Lệnh khác

```bash
make help
make shell
make clean
make down
make check-gpu
```

---

## Log tài nguyên khi train

Sau `make train`, script in thêm block `RESOURCE USAGE` để xem lần train vừa rồi dùng bao nhiêu CPU/RAM/GPU.

```text
╭────────────────────────────────────────────────────╮
│ RESOURCE USAGE                                     │
├────────────────────────────────────────────────────┤
│ CPU available    8 cores                           │
│ CPU avg          1.25 cores (15.6%)                │
│ CPU peak         2.40 cores (30.0%)                │
│ RAM peak         720.0 / 4096 MB (17.6%)           │
│ VRAM base        320 MB                            │
│ VRAM peak        2100 / 8192 MB                    │
│ VRAM delta       1780 MB                           │
│ GPU util avg     35.4%                             │
│ GPU util peak    82%                               │
╰────────────────────────────────────────────────────╯
```

| Chỉ số | Ý nghĩa |
| ------ | ------- |
| `CPU available` | Số CPU core container có thể dùng. Nếu Docker bị giới hạn CPU thì số này là quota, không nhất thiết bằng toàn bộ core của VPS. |
| `CPU avg` | Trung bình trong cả quá trình train dùng bao nhiêu core. Ví dụ `1.25 cores (15.6%)` nghĩa là trung bình chỉ dùng khoảng 1.25/8 core. |
| `CPU peak` | Mức CPU cao nhất bắt được trong lúc train. Nếu gần 100% lâu dài thì CPU đang là nút thắt. |
| `RAM peak` | RAM cao nhất process train dùng. Nếu có dạng `720 / 4096 MB` thì vế sau là RAM limit/total đọc được trong container. |
| `VRAM base` | VRAM đã dùng ở thời điểm bắt đầu đo. Có thể gồm CUDA context hoặc process khác đang dùng GPU. |
| `VRAM peak` | VRAM cao nhất trong lúc train, kèm tổng VRAM GPU. Dùng để chọn card có đủ VRAM. |
| `VRAM delta` | Phần VRAM tăng thêm do lần train này so với baseline. |
| `GPU util avg` | Mức sử dụng GPU trung bình. Thấp nghĩa là GPU thường phải chờ CPU/data/model quá nhỏ. |
| `GPU util peak` | Mức sử dụng GPU cao nhất bắt được. |

Gợi ý đọc nhanh: nếu `GPU util avg` thấp nhưng `CPU avg/peak` cao, nên tăng CPU hoặc tối ưu pipeline data; nếu `VRAM peak` sát tổng VRAM thì cần giảm batch/config hoặc dùng GPU nhiều VRAM hơn; nếu cả CPU/GPU đều thấp thì bài toán/model hiện tại có thể quá nhỏ nên GPU không nhanh hơn CPU nhiều.

---

## Lỗi thường gặp

| Lỗi | Cách xử lý |
| --- | ---------- |
| `Cannot connect to Docker` | Mở Docker Desktop |
| `Chưa có model tại models/cpu/...` | `make preprocess && make train` |
| `[WARN] bỏ qua entity ... offset sai` | Kiểm tra `train_data.py` / `dev_data.py` |
| Train ~1–3 phút | Bình thường (`max_steps` = 20.000) |

---

## File quan trọng

| File | Vai trò |
| ---- | ------- |
| `data/train_data.py` | Thêm câu train (`_make()`) |
| `data/dev_data.py` | Câu dev — không trùng train |
| `data/test_data.py` | 25 query thử sau train |
| `models/cpu/`, `models/gpu/` | Model sau train trong container — dùng `model-best/` |
| `config/config.cfg` | Cài đặt train — [config/GIAI_THICH.md](config/GIAI_THICH.md) |
| `docs/training_data_spec.md` | Đặc tả nhãn và pattern data |
| `docs/env.md` | Biến `.env` + `docker-compose.gpu.yml` |

---

## Luồng

```
train_data.py ──preprocess──► train.spacy + dev.spacy ──train──► models/{cpu|gpu}/model-best
                                                                        │
                                                                   predict
                                                                        ▼
                                                              CATEGORY, PRODUCT, SPEC
```

---

## Dùng model ở backend

Copy `models/cpu/model-best/` (hoặc `gpu/`) sang server:

```python
import re
import spacy

PRICE_PATTERNS = [
    r"dưới\s+\d+[k|tr|triệu|đồng]*",
    r"từ\s+\d+[k|tr]*\s+đến\s+\d+[k|tr|triệu]*",
    r"giá\s+(rẻ|tốt|cao)",
    r"(giảm giá|khuyến mãi|sale|flash sale)",
]
STOPWORDS = {"mua", "tìm", "bán", "còn", "có", "không", "xem", "cho", "tôi"}

def preprocess(query: str):
    price_matches = []
    for pat in PRICE_PATTERNS:
        for m in re.finditer(pat, query, re.I):
            price_matches.append(m.group())
            query = query.replace(m.group(), " ")
    tokens = [t for t in query.split() if t.lower() not in STOPWORDS]
    return " ".join(tokens), price_matches

nlp = spacy.load("path/to/model-best")

def parse(query: str):
    clean, prices = preprocess(query)
    doc = nlp(clean)
    entities = [{"text": e.text, "label": e.label_} for e in doc.ents]
    return {"entities": entities, "prices": prices}
```

---

## Thêm data

1. Thêm câu vào `train_data.py` và `dev_data.py` (dùng `_make()`).
2. `make build && make preprocess && make train`
3. `make test-predict`

Giữ ~80% train / 20% dev. Không annotate giá / stopword — thuộc tầng logic.

---

## Tài liệu

- [docs/env.md](docs/env.md) — biến `.env` (4 biến)
- [docs/training_data_spec.md](docs/training_data_spec.md) — đặc tả training data
- [config/GIAI_THICH.md](config/GIAI_THICH.md) — config train

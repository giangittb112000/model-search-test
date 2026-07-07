# Đặc tả Kỹ thuật — Tập Training Data cho NER Search Model

**Dự án:** Harness Search — spaCy NER
**Đối tượng đọc:** người annotate data, ML engineer, reviewer
**Tham chiếu:** `docs/data_ex/categories.json`, `docs/data_ex/products.json`, `docs/data_ex/specs.json`

---

## 1. Mục tiêu tập training

Xây dựng tập câu query **giống hành vi user thật** đủ để spaCy học được cách bóc **CATEGORY / PRODUCT / SPEC** từ query search e-commerce tiếng Việt. Tối ưu cho **query ngắn (1–8 từ)** với nhiều tổ hợp entity — không tối ưu cho câu văn hoàn chỉnh.

**Scope model:** chỉ 3 nhãn `CATEGORY`, `PRODUCT`, `SPEC`.
**Không thuộc scope model:** PRICE và STOPWORD — xử lý bằng logic (regex + word list) ở tầng application. Xem §2.3 bên dưới.

---

## 2. Schema nhãn

### 2.1. Bảng nhãn chính thức

| Label | Ý nghĩa | Downstream sử dụng |
|-------|---------|---------------------|
| `CATEGORY` | Danh mục hàng hóa | ES `match` field `category` |
| `PRODUCT` | Tên sản phẩm / dòng máy / **thương hiệu** | ES `match` field `product_name`, `brand` |
| `SPEC` | Thông số kỹ thuật, thuộc tính | ES `filter` term theo `key` spec |

### 2.2. Định nghĩa chi tiết

#### CATEGORY
- **Là gì:** loại hàng chung, không gắn với hãng cụ thể.
- **Nguồn:** `categories.json` — các name có `level: 2` và các name mang tính "loại hàng" ở `level: 3` (`Máy giặt`, `Điện lạnh`, `Máy ép chậm`).
- **Ví dụ hợp lệ:** `laptop`, `điện thoại`, `tai nghe`, `máy giặt`, `ốp lưng`, `bàn phím cơ`, `flycam`, `loa`.
- **Không phải CATEGORY:** `iPad 9.7` (là PRODUCT/model), `Corsair` (là PRODUCT/brand).

#### PRODUCT
- **Là gì:** tên hãng, dòng máy, model cụ thể, mã sản phẩm.
- **Nguồn:**
  - `products.json` — trích subset từ tên đầy đủ.
  - `specs.json` — trường `phone_accessory_brands` (Anker, ESR, KST), `for_product` (iPad Air | Pro 10.5).
  - `categories.json` — các name là dòng máy (`iPad mini`, `Galaxy Tab 10.1`, `TCL`).
- **Ví dụ hợp lệ:** `iphone 15`, `dell xps`, `sandisk`, `anker`, `dji avata`, `galaxy a37`.
- **Quy tắc gộp:** brand + model liền nhau → gom vào 1 span `PRODUCT` duy nhất (`samsung galaxy s24` là 1 span).

#### SPEC
- **Là gì:** thông số kỹ thuật (RAM, dung lượng, kích thước, công suất), thuộc tính (chính hãng, cũ, mới), tính năng.
- **Nguồn:** `specs.json` — các value `basic[]` và `full_by_group[]`.
- **Ví dụ hợp lệ:**
  - Con số + đơn vị: `128 gb`, `13 inch`, `65w`, `200mb/s`.
  - Trạng thái: `chính hãng`, `cũ`, `mới`, `body only`, `combo`.
  - Cổng/chuẩn: `usb-c`, `bluetooth`, `5g`, `wifi 6`.
  - Đặc tính: `bao da`, `chống nước`, `có chân đứng`.
- **Quy tắc gộp:** số + đơn vị đi liền → 1 span (`128gb`, `128 gb`, `13 inch` đều là 1 span SPEC).

### 2.3. Các thành phần KHÔNG annotate

Không gắn nhãn cho các từ/cụm sau — cứ để trống, model bỏ qua tự nhiên:

| Loại | Ví dụ | Xử lý ở đâu |
|------|-------|-------------|
| Giá / khuyến mãi | `giá rẻ`, `dưới 10 triệu`, `từ 500k đến 1 triệu`, `giảm giá` | Regex trước khi gọi NER |
| Stopword | `mua`, `tìm`, `bán`, `còn`, `có`, `không`, `cho tôi`, `xem` | Word list, strip trước NER |
| Từ đệm | `tại`, `ở`, `nhé`, `dùm` | Word list, strip trước NER |

---

## 3. Quy tắc xử lý ca biên

| Tình huống | Quy tắc | Ví dụ |
|-----------|---------|-------|
| CATEGORY và PRODUCT trùng nhau | Ưu tiên PRODUCT nếu có số/model | `ipad` → CATEGORY; `ipad mini` → PRODUCT |
| Brand đi lẻ | Vẫn là PRODUCT | `sandisk` → PRODUCT |
| Brand có model số | Gom cả cụm | `iphone 15 pro max` → 1 span PRODUCT |
| SPEC dạng dính | Vẫn là 1 span | `256gb`, `27inch` → 1 span SPEC |
| Nhiều SPEC liền kề | Tách từng span | `8gb 256gb` → 2 span SPEC riêng |
| Từ không thuộc nhãn nào | **Không** label | `tại`, `ở`, `nhé`, `mua`, `giá rẻ` |
| Câu chỉ có giá / stopword | **Bỏ khỏi tập** | `giá rẻ`, `mua hàng` — model không cần học |
| Câu chỉ có 1 từ vô nghĩa | **Bỏ khỏi tập** | `abcxyz` |

---

## 4. Format Data

### 4.1. Định dạng lưu trữ

Dùng helper `_make()` — mỗi token là `(text, label|None)`, offset tự tính:

```python
from data.train_data import _make

C, P, S = "CATEGORY", "PRODUCT", "SPEC"

TRAIN_DATA = [
    _make([("laptop", C), ("dell xps", P), ("16gb", S)]),
    _make([("iphone 15", P), ("chính hãng", S)]),
]
```

Không tự tính offset thủ công (tránh sai `start`/`end`).

### 4.2. Quy tắc offset (khi cần ghi tay)

- `start` = index ký tự đầu của span.
- `end` = index ký tự **cuối + 1** (Python slice).
- `text[start:end]` phải bằng đúng cụm.
- **Không overlap**, không space thừa đầu/cuối.

### 4.3. Auto-verify

`train_data.py` và `dev_data.py` chạy `_verify()` khi import — fail sớm nếu offset sai hoặc dùng label ngoài `{CATEGORY, PRODUCT, SPEC}`.

---

## 5. Cân đối tập data

### 5.1. Kích thước tối thiểu

| Tập | Số câu | Ghi chú |
|-----|--------|---------|
| Train | **~150** | Đủ đa dạng 3 nhãn × 4 pattern |
| Dev | **~35** | Không trùng train |
| Test (predict thử) | **~25** | Query hoàn toàn mới |

### 5.2. Phân phối theo pattern

| Pattern | % target train | Ví dụ |
|---------|---------------|-------|
| P1: chỉ CATEGORY | ~20% | `laptop` |
| P2: chỉ PRODUCT | ~20% | `iphone 15` |
| P3: CATEGORY + PRODUCT | ~27% | `laptop dell` |
| P4: CATEGORY + PRODUCT + SPEC | ~33% | `laptop dell 16gb` |

### 5.3. Phân phối theo nhãn (đếm span)

| Label | % target | Lý do |
|-------|----------|-------|
| CATEGORY | ~40% | Xuất hiện ở P1, P3, P4 |
| PRODUCT | ~40% | Xuất hiện ở P2, P3, P4 |
| SPEC | ~20% | Chỉ ở P4 nhưng đa dạng đơn vị/keyword |

### 5.4. Chống bias vị trí

- **KHÔNG** để >50% câu có CATEGORY ở đầu.
- Thêm câu có:
  - PRODUCT ở đầu: `iphone 15 chính hãng`.
  - SPEC ở đầu: `128gb sandisk`.
  - Nhiều SPEC ở cuối để model học kết hợp.

---

## 6. Nguồn sinh data

### 6.1. Từ `categories.json`
- Name → CATEGORY: `Máy giặt`, `Máy ép chậm`, `Điện lạnh`, `Máy lọc nước`…
- Name → PRODUCT (dòng máy/brand): `iPad mini`, `Galaxy Tab 10.1`, `TCL`, `Tecno`…
- Lowercase khi đưa vào train.

### 6.2. Từ `products.json`
- **Không** copy nguyên title. Trích **subset thực tế** user gõ:
  - `Samsung Galaxy A37 8GB 256GB` → `galaxy a37`, `galaxy a37 8gb`, `samsung a37 256gb`.
  - `Loa JBL Partybox Stage 330` → `loa jbl`, `jbl partybox`, `loa jbl partybox 330`.
- Mỗi tên catalog sinh 2–4 query training.

### 6.3. Từ `specs.json`
- Kết hợp `phone_accessory_brands` (PRODUCT) + `for_product` (PRODUCT) + spec value (SPEC):
  - `ESR` + `iPad Air 10.5` + `Bao da` → `ốp esr ipad air 10.5 bao da`.
  - `Anker` + `65W` + `USB-C` → `sạc anker 65w usb-c`.
- Đây là nguồn chính cho pattern P4.

### 6.4. Từ search history (nếu có)
- Top head queries → P1/P2.
- Long-tail queries zero-result → annotate → P4.

---

## 7. Quy trình xây dựng

```
1. Chuẩn bị vocabulary từ 3 file JSON trong `docs/data_ex/`

2. Viết TRAIN_DATA theo pattern P1–P4 (target ~150 câu)
   ├─ Copy từ catalog (subset)
   └─ Trộn từ search history

3. Chia dev set (không trùng train, ~35 câu, phủ đủ 3 nhãn)

4. Auto-verify offset khi import (đã tích hợp)

5. make preprocess → tạo .spacy binary

6. make train → xem F-score dev

7. make test-predict → kiểm tra định tính

8. Iterate:
   ├─ F < 70%: thêm mẫu, cân bằng bias
   ├─ Nhãn X F thấp: tăng ví dụ nhãn X
   └─ Nhầm lẫn A↔B: thêm mẫu phân biệt rõ
```

---

## 8. Quality Checklist trước khi merge

- [ ] `_verify()` pass (auto khi import).
- [ ] Không câu dev nào trùng train.
- [ ] Mỗi nhãn xuất hiện ≥ 20 lần trong train.
- [ ] Có ≥ 15 câu cho mỗi pattern P1–P4.
- [ ] Không câu > 12 từ.
- [ ] Toàn bộ text lowercase (trừ mã sản phẩm ký tự đặc biệt).
- [ ] Không tab, double space, trailing space.
- [ ] Ít nhất 60% câu có > 1 entity.
- [ ] **Không** annotate cho `mua`, `tìm`, `giá rẻ`, `dưới N triệu`…

---

## 9. Ví dụ mẫu

```python
from data.train_data import _make

C, P, S = "CATEGORY", "PRODUCT", "SPEC"

TRAIN_DATA = [
    # P1: chỉ CATEGORY
    _make([("laptop", C)]),
    _make([("máy giặt", C)]),

    # P2: chỉ PRODUCT
    _make([("iphone 15", P)]),
    _make([("sandisk", P)]),

    # P3: CATEGORY + PRODUCT
    _make([("laptop", C), ("dell", P)]),
    _make([("ốp lưng", C), ("iphone 15", P)]),

    # P4: CATEGORY + PRODUCT + SPEC
    _make([("laptop", C), ("dell xps", P), ("16gb", S)]),
    _make([("sạc", C), ("anker", P), ("65w", S)]),
    _make([("ốp lưng", C), ("ipad air", P), ("bao da", S)]),
]
```

---

## 10. Bảo trì & Iteration

- **Mỗi lần thêm entity mới** vào catalog → thêm ≥ 3 query mẫu vào train.
- **Mỗi tháng** review log query zero-result → annotate → bổ sung.
- **Không** tăng đột ngột 1 pattern → gây bias.
- **Version data** như code: commit rõ ràng.

---

## 11. Roadmap theo kích thước data

| Kích thước train | Bổ sung | Target F-score dev |
|-----------------|---------|--------------------|
| ~150 câu | Baseline 3 nhãn, 4 pattern | ≥ 85% |
| ~300 câu | Bổ sung từ zero-result log | ≥ 88% |
| ~500 câu | Typo, không dấu, spec đa dạng | ≥ 90% |
| ~1000 câu | Cân nhắc thêm intent classifier (khám phá / mua) | ≥ 92% |

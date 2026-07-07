# Biến môi trường (`.env`)

Tạo file: `make env-init-cpu` hoặc `make env-init-gpu`.

---

## 4 biến

| Biến | Giá trị | Ý nghĩa |
| ---- | ------- | ------- |
| **`NER_TRAIN_MODE`** | `cpu` / `gpu` | Train trên CPU hay GPU → `models/cpu/` hoặc `models/gpu/` |
| **`NER_RUN_MODE`** | `cpu` / `gpu` | Chạy predict trên CPU hay GPU |
| **`NER_GPU_ID`** | `0`, `1`, … | Card NVIDIA thứ mấy (`nvidia-smi`) |
| **`NER_GPU_ALLOCATOR`** | xem bên dưới | Cách spaCy **chia bộ nhớ GPU** khi train (không chọn CPU/GPU) |

Tự động suy ra:

| Việc | Khi nào |
| ---- | ------- |
| Merge `docker-compose.gpu.yml` | `NER_TRAIN_MODE=gpu` hoặc `NER_RUN_MODE=gpu` |
| `spacy train --gpu-id` | `-1` nếu train cpu; `NER_GPU_ID` nếu train gpu |
| Folder model predict | `models/{NER_TRAIN_MODE}/model-best` |

---

## `NER_GPU_ALLOCATOR` là gì?

Theo [spaCy config](https://spacy.io/api/data-formats#config): đây là setting **`[system].gpu_allocator`** — *“library for CuPy to route GPU memory allocation to”*.

**Không phải** nút bật CPU/GPU. Việc chọn card train là **`--gpu-id`** (map từ `NER_TRAIN_MODE` + `NER_GPU_ID`).

### Các giá trị hợp lệ

| Giá trị | Khi nào dùng |
| ------- | ------------ |
| **`null`** | Train **CPU**, hoặc train GPU nhưng không route qua PyTorch/TensorFlow (mặc định an toàn) |
| **`pytorch`** | Train **GPU** và image có **PyTorch + CUDA** — khuyến nghị cho project này |
| **`tensorflow`** | Train **GPU** và stack dùng **TensorFlow** thay PyTorch |
| **`auto`** | Tương đương `null` trong image mặc định |

### `pytorch` và `tensorflow` nghĩa là gì?

Khi train trên GPU, spaCy/Thinc dùng **CuPy** để tính toán trên card. Nếu trong cùng process còn **PyTorch** hoặc **TensorFlow** (custom model, transformer, v.v.), mỗi framework có **pool bộ nhớ GPU riêng** → dễ **OOM** (hết VRAM) dù còn chỗ trống.

`gpu_allocator` bảo CuPy: *“xin bộ nhớ qua pool của PyTorch (hoặc TensorFlow)”* — **một pool chung**, ít tràn bộ nhớ hơn.

Project NER hiện tại dùng kiến trúc Thinc mặc định (tok2vec + ner), **không bắt buộc** `pytorch` để chạy GPU. Image mặc định chỉ cài CuPy cho GPU, nên dùng **`null`**. Chỉ đặt **`pytorch`** khi bạn đã cài thêm PyTorch+CUDA vào image.

**Predict** (`make predict`) **không** dùng `gpu_allocator` — chỉ `spacy.require_gpu()` khi `NER_RUN_MODE=gpu`.

---

## Ví dụ `.env`

**Máy CPU (mặc định):**

```env
NER_TRAIN_MODE=cpu
NER_RUN_MODE=cpu
NER_GPU_ID=0
NER_GPU_ALLOCATOR=null
```

**Train GPU + predict CPU:**

```env
NER_TRAIN_MODE=gpu
NER_RUN_MODE=cpu
NER_GPU_ID=0
NER_GPU_ALLOCATOR=null
```

**Train + predict GPU:**

```env
NER_TRAIN_MODE=gpu
NER_RUN_MODE=gpu
NER_GPU_ID=0
NER_GPU_ALLOCATOR=null
```

---

## NVIDIA / card khác

Stack Docker hiện tại: **NVIDIA + CUDA**. AMD/Intel/Mac GPU → giữ `NER_TRAIN_MODE=cpu`, `NER_RUN_MODE=cpu`.

Image mặc định dùng `nvidia/cuda:12.4.1-runtime-ubuntu22.04` + `cupy-cuda12x` + `nvidia-cuda-runtime-cu12` (chứa CUDA headers cho CuPy NVRTC JIT — không có sẽ báo `cannot open source file "cuda_fp16.h"`).

## Docker GPU

`NER_TRAIN_MODE=gpu` hoặc `NER_RUN_MODE=gpu` → Makefile merge `docker-compose.gpu.yml`. Cần [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Verify sau `make up`: `make check-gpu` — script sẽ trigger NVRTC compile một kernel nhỏ để phát hiện thiếu headers ngay.

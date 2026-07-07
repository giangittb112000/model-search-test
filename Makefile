SHELL := /bin/bash

ifneq (,$(wildcard .env))
include .env
export
endif

NER_TRAIN_MODE ?= cpu
NER_RUN_MODE ?= cpu

DC_BASE := docker compose -f docker-compose.yml
DC_GPU  := $(DC_BASE) -f docker-compose.gpu.yml
USE_GPU_COMPOSE := false

ifeq ($(NER_TRAIN_MODE),gpu)
USE_GPU_COMPOSE := true
endif

ifeq ($(NER_RUN_MODE),gpu)
USE_GPU_COMPOSE := true
endif

ifeq ($(USE_GPU_COMPOSE),true)
DC := $(DC_GPU)
else
DC := $(DC_BASE)
endif

RUN := $(DC) exec nlp

Q ?= mua iphone 15 giá rẻ

.PHONY: help require-env env-init env-init-cpu env-init-gpu build up down preprocess train predict test-predict check-gpu _check_gpu_run shell clean

require-env:
	@test -f .env || ( \
		echo "Thiếu .env — chạy: make env-init-cpu hoặc make env-init-gpu"; \
		exit 1)

help:
	@echo "Targets (cần .env — 4 biến: docs/env.md):"
	@echo "  env-init-cpu / env-init-gpu   tạo .env từ mẫu"
	@echo "  build                         build image (COPY src/data/config vào image)"
	@echo "  up / down                     start/stop container (giữ chạy nền)"
	@echo "  preprocess                    data/*.py -> data/*.spacy trong container"
	@echo "  train                         train (đọc NER_* trong .env)"
	@echo "  predict Q='...'               predict 1 câu"
	@echo "  test-predict                  predict TEST_QUERIES"
	@echo "  check-gpu                     verify CuPy + GPU trong container"
	@echo "  shell / clean"
	@echo ""
	@echo "Ghi chú: source/data được COPY vào image lúc build."
	@echo "  Đổi code hoặc data → 'make build' rồi 'make down && make up'."

env-init-cpu:
	cp .env.cpu.example .env

env-init-gpu:
	cp .env.gpu.example .env

env-init: env-init-cpu

build: require-env
	$(DC_BASE) build

up: require-env
	$(DC) up -d

down:
	$(DC) down

preprocess: require-env up
	$(RUN) python src/preprocess.py

train: require-env up
	$(RUN) python src/train.py

predict: require-env up
	$(RUN) python src/predict.py "$(Q)"

test-predict: require-env up
	$(RUN) python src/test_predict.py

check-gpu: require-env
	@if [ "$(NER_TRAIN_MODE)" != "gpu" ] && [ "$(NER_RUN_MODE)" != "gpu" ]; then \
		echo "[WARN] .env đang CPU mode — container không bind GPU, check-gpu sẽ fail."; \
		echo "        Đặt NER_TRAIN_MODE=gpu hoặc NER_RUN_MODE=gpu rồi 'make up' lại."; \
	fi
	$(MAKE) --no-print-directory _check_gpu_run

_check_gpu_run: up
	$(RUN) python -c "\
import spacy, cupy, thinc.compat as c; \
spacy.require_gpu(0); \
print('spaCy', spacy.__version__, '| CuPy', cupy.__version__); \
print('has_cupy', c.has_cupy, '| has_gpu', c.has_gpu, '| devices', cupy.cuda.runtime.getDeviceCount()); \
x = cupy.array([1., 2., 3.], dtype=cupy.float32); \
y = (x + x).sum().item(); \
print('NVRTC compile OK — kernel result:', y)"

shell: require-env up
	$(RUN) bash

clean: require-env up
	$(RUN) rm -rf models data/train.spacy data/dev.spacy

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

.PHONY: help require-env env-init env-init-cpu env-init-gpu build up down preprocess train predict test-predict check-gpu shell clean

require-env:
	@test -f .env || ( \
		echo "Thiếu .env — chạy: make env-init-cpu hoặc make env-init-gpu"; \
		exit 1)

help:
	@echo "Targets (cần .env — 4 biến: docs/env.md):"
	@echo "  env-init-cpu / env-init-gpu"
	@echo "  build | up | down | preprocess | train | predict | test-predict"
	@echo "  check-gpu | shell | clean"

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

check-gpu: require-env up
	$(RUN) python -c "import spacy, cupy, thinc.compat as c; spacy.require_gpu(0); print('spaCy', spacy.__version__); print('CuPy', cupy.__version__); print('has_cupy', c.has_cupy); print('has_gpu', c.has_gpu); print('CUDA devices', cupy.cuda.runtime.getDeviceCount())"

shell: require-env up
	$(RUN) bash

clean: require-env up
	$(RUN) rm -rf models data/train.spacy data/dev.spacy

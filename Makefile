SHELL := /bin/bash

ifneq (,$(wildcard .env))
include .env
export
endif

NER_TRAIN_MODE ?= cpu
NER_RUN_MODE ?= cpu

DC_BASE := UID=$$(id -u) GID=$$(id -g) docker compose -f docker-compose.yml
DC_GPU  := $(DC_BASE) -f docker-compose.gpu.yml

ifeq ($(NER_TRAIN_MODE),gpu)
RUN_TRAIN := $(DC_GPU) run --rm
else
RUN_TRAIN := $(DC_BASE) run --rm
endif

ifeq ($(NER_RUN_MODE),gpu)
RUN_PREDICT := $(DC_GPU) run --rm
else
RUN_PREDICT := $(DC_BASE) run --rm
endif

DC  := $(DC_BASE)
RUN := $(DC_BASE) run --rm

Q ?= mua iphone 15 giá rẻ

.PHONY: help require-env build preprocess init-config train train-gpu predict test-predict shell clean env-init env-init-cpu env-init-gpu

require-env:
	@test -f .env || ( \
		echo "Thiếu .env — chạy: make env-init-cpu hoặc make env-init-gpu"; \
		exit 1)

help:
	@echo "Targets (cần .env — 4 biến: docs/env.md):"
	@echo "  env-init-cpu / env-init-gpu"
	@echo "  build | preprocess | train | predict | test-predict | shell | clean"
	@echo "  train-gpu      - train khi NER_TRAIN_MODE=gpu trong .env"

env-init-cpu:
	cp .env.cpu.example .env

env-init-gpu:
	cp .env.gpu.example .env

env-init: env-init-cpu

build: require-env
	$(DC) build

preprocess: require-env
	$(RUN) nlp python src/preprocess.py

init-config: require-env
	$(RUN) nlp python -m spacy init config config/config.cfg --lang vi --pipeline ner --optimize efficiency --force

train: require-env
	$(RUN_TRAIN) nlp python src/train.py

train-gpu: require-env
	@grep -qE '^NER_TRAIN_MODE=gpu' .env || ( \
		echo "Đặt NER_TRAIN_MODE=gpu trong .env (hoặc make env-init-gpu)"; exit 1)
	$(RUN_TRAIN) nlp python src/train.py

predict: require-env
	$(RUN_PREDICT) nlp python src/predict.py "$(Q)"

test-predict: require-env
	$(RUN_PREDICT) nlp python src/test_predict.py

shell: require-env
	$(RUN) nlp bash

clean:
	rm -rf models data/train.spacy data/dev.spacy

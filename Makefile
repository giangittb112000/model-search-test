SHELL := /bin/bash

ifneq (,$(wildcard .env))
include .env
export
endif

NER_TRAIN_MODE ?= cpu
NER_RUN_MODE ?= cpu

DC_BASE := docker compose -f docker-compose.yml
DC_GPU  := $(DC_BASE) -f docker-compose.gpu.yml

ifeq ($(NER_TRAIN_MODE),gpu)
DC_TRAIN := $(DC_GPU)
else
DC_TRAIN := $(DC_BASE)
endif

ifeq ($(NER_RUN_MODE),gpu)
DC_PREDICT := $(DC_GPU)
else
DC_PREDICT := $(DC_BASE)
endif

DC  := $(DC_BASE)
RUN := $(DC_BASE) exec nlp
RUN_TRAIN := $(DC_TRAIN) exec nlp
RUN_PREDICT := $(DC_PREDICT) exec nlp

Q ?= mua iphone 15 giá rẻ

.PHONY: help require-env build up up-train up-predict down preprocess init-config train train-gpu predict test-predict shell clean env-init env-init-cpu env-init-gpu

require-env:
	@test -f .env || ( \
		echo "Thiếu .env — chạy: make env-init-cpu hoặc make env-init-gpu"; \
		exit 1)

help:
	@echo "Targets (cần .env — 4 biến: docs/env.md):"
	@echo "  env-init-cpu / env-init-gpu"
	@echo "  build | up | down | preprocess | train | predict | test-predict | shell | clean"
	@echo "  train-gpu      - train khi NER_TRAIN_MODE=gpu trong .env"

env-init-cpu:
	cp .env.cpu.example .env

env-init-gpu:
	cp .env.gpu.example .env

env-init: env-init-cpu

build: require-env
	$(DC) build

up: require-env
	$(DC) up -d

up-train: require-env
	$(DC_TRAIN) up -d

up-predict: require-env
	$(DC_PREDICT) up -d

down:
	$(DC) down

preprocess: require-env up
	$(RUN) python src/preprocess.py

init-config: require-env up
	$(RUN) python -m spacy init config config/config.cfg --lang vi --pipeline ner --optimize efficiency --force

train: require-env up-train
	$(RUN_TRAIN) python src/train.py

train-gpu: require-env
	@grep -qE '^NER_TRAIN_MODE=gpu' .env || ( \
		echo "Đặt NER_TRAIN_MODE=gpu trong .env (hoặc make env-init-gpu)"; exit 1)
	$(DC_TRAIN) up -d
	$(RUN_TRAIN) python src/train.py

predict: require-env up-predict
	$(RUN_PREDICT) python src/predict.py "$(Q)"

test-predict: require-env up-predict
	$(RUN_PREDICT) python src/test_predict.py

shell: require-env up
	$(RUN) bash

clean: require-env up
	$(RUN) rm -rf models data/train.spacy data/dev.spacy

# Every target is a thin wrapper over `python -m onegap.cli <cmd>`, so the same command runs
# locally on Windows (`uv run onegap <cmd>`) and inside HF Jobs / CI (`make <target>`).
PY ?= python
CLI = $(PY) -m onegap.cli
ARGS ?=

.PHONY: test lint check-env smoke data translate codemix benign freeze verify-frozen \
        run judge metrics report prefdata circuits train-D train-E train-R steer-C \
        edit-size replay eval-test

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

check-env:      ; $(CLI) check-env $(ARGS)
smoke:          ; $(CLI) smoke $(ARGS)
data:           ; $(CLI) data $(ARGS)
translate:      ; $(CLI) translate $(ARGS)
codemix:        ; $(CLI) codemix $(ARGS)
benign:         ; $(CLI) benign $(ARGS)
freeze:         ; $(CLI) freeze $(ARGS)
verify-frozen:  ; $(CLI) verify-frozen $(ARGS)
run:            ; $(CLI) run $(ARGS)
judge:          ; $(CLI) judge $(ARGS)
metrics:        ; $(CLI) metrics $(ARGS)
report:         ; $(CLI) report $(ARGS)
prefdata:       ; $(CLI) prefdata $(ARGS)
circuits:       ; $(CLI) circuits $(ARGS)
circuit-check:  ; $(CLI) circuit-check $(ARGS)
train-D:        ; $(CLI) train --config configs/train_D.yaml $(ARGS)
train-E:        ; $(CLI) train --config configs/train_E.yaml $(ARGS)
train-R:        ; $(CLI) train --config configs/train_R.yaml $(ARGS)
steer-C:        ; $(CLI) steer --config configs/steer_C.yaml $(ARGS)
edit-size:      ; $(CLI) edit-size $(ARGS)
replay:         ; $(CLI) replay $(ARGS)
eval-test:      ; $(CLI) verify-frozen && $(CLI) run --split test $(ARGS)

curate:         ; $(CLI) curate $(ARGS)
vllm-smoke:     ; $(CLI) vllm-smoke $(ARGS)
score-hinglish:  ; $(CLI) score-hinglish $(ARGS)
codemix-back:    ; $(CLI) codemix-back $(ARGS)
codemix-regen:   ; $(CLI) codemix-regen $(ARGS)
score-hinglish-back: ; $(CLI) score-hinglish-back $(ARGS)
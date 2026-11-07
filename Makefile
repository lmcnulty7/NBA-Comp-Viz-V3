# Makefile: the Mac run order in one place. `make help` lists the targets; docs/run_order.md has the long form.
# Variables: PY (interpreter), DATA (real DATA_ROOT, the parent of sportvu_contract/ and collect/; unset means
# fixtures), OUT (OUT_ROOT for run_all), DATA_ROOT (command line only: where `make fixtures` writes and `make phase0` reads when
# DATA is unset), SITE_DIR (where `make site` copies the page), SKIP_HELDOUT (1, or any value other than empty or 0,
# passes --skip-heldout to every train step through run_all: held-out games set aside but never scored, the rule
# for every real run before stage 8).
# Every path is quoted: Drive paths have spaces.

PY ?= python3
DATA ?=
OUT ?=
DATA_ROOT ?= .fixture_root
# docs/mac_runbook.md exports DATA_ROOT (the real Drive folder) in ~/.zshrc. That environment value must never become
# the fixture root, or `make fixtures` would write synthetic data into the real dataset; only DATA_ROOT=... given on
# the make command line moves it.
ifeq ($(origin DATA_ROOT),environment)
DATA_ROOT := .fixture_root
endif
SITE_DIR ?= site
PYTEST ?= $(PY) -m pytest -q
SKIP_HELDOUT ?=
SKIP_ARGS = $(if $(filter-out 0,$(SKIP_HELDOUT)), --skip-heldout,)
RUN_ARGS = $(if $(DATA),--data "$(DATA)",--fixtures)$(if $(OUT), --out "$(OUT)",)$(SKIP_ARGS)

EXTENSIONS = xfg rebounds matchups gravity ghosting embeddings epv actions coverage decisions playcalling lineups winprob load betting

.PHONY: help test test-fast fixtures phase0 all all-fixtures collect-bref site clean-logs list $(addprefix ext-,$(EXTENSIONS))

help: ## list the targets with their one-line descriptions
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/^([a-zA-Z0-9_-]+):.*## (.*)$$/  \1\t\2/' | sort | expand -t 18
	@echo
	@echo "  make ext-xfg DATA=/path/to/DATA_ROOT runs on real data; make ext-xfg alone runs on fixtures."
	@echo "  SKIP_HELDOUT=1 on ext-*, all or all-fixtures passes --skip-heldout to every train step (every real run before stage 8);"
	@echo "  any value other than empty or 0 turns it on."
	@echo "  Lanes read the Phase 0 tables and states_1hz from OUT first, then DATA; leave OUT unset (OUT_ROOT = DATA_ROOT) for the simplest setup."

test: ## run every test under tests/
	$(PYTEST) tests

test-fast: ## run the tests not marked slow
	$(PYTEST) -m "not slow" tests

fixtures: ## write a synthetic fixture dataset to DATA_ROOT (default .fixture_root) with analytics.fixtures
	$(PY) -c "import sys; from analytics.fixtures import write_fixture_root; write_fixture_root(sys.argv[1], seed=0, n_games=4, heldout_every=4)" "$(DATA_ROOT)"
	@echo "fixture root: $(DATA_ROOT)"

phase0: ## run analytics.gaps then analytics.run_phase0 on DATA (or DATA_ROOT when DATA is unset)
	$(PY) run_all.py --phase0 --data "$(or $(DATA),$(DATA_ROOT))"$(if $(OUT), --out "$(OUT)",)

list: ## print the extension registry with availability
	$(PY) run_all.py --list

ext-xfg: ## xfg: validate, features, train (DATA=/path for real data, else fixtures)
	$(PY) run_all.py $(RUN_ARGS) --ext xfg

ext-rebounds: ## rebounds: validate, features, train
	$(PY) run_all.py $(RUN_ARGS) --ext rebounds

ext-matchups: ## matchups: validate, features, train (depends on xfg)
	$(PY) run_all.py $(RUN_ARGS) --ext matchups

ext-gravity: ## gravity: validate, features, train
	$(PY) run_all.py $(RUN_ARGS) --ext gravity

ext-ghosting: ## ghosting: validate, features, train
	$(PY) run_all.py $(RUN_ARGS) --ext ghosting

ext-embeddings: ## embeddings: validate, features, train
	$(PY) run_all.py $(RUN_ARGS) --ext embeddings

ext-epv: ## epv: validate, states, features, train
	$(PY) run_all.py $(RUN_ARGS) --ext epv

ext-actions: ## actions: validate, features, train
	$(PY) run_all.py $(RUN_ARGS) --ext actions

ext-coverage: ## coverage: validate, features, train (depends on actions)
	$(PY) run_all.py $(RUN_ARGS) --ext coverage

ext-decisions: ## decisions: validate, features, train (depends on xfg, epv)
	$(PY) run_all.py $(RUN_ARGS) --ext decisions

ext-playcalling: ## playcalling: validate, features, train (depends on epv, actions, coverage)
	$(PY) run_all.py $(RUN_ARGS) --ext playcalling

ext-lineups: ## lineups: validate, stints, features, train
	$(PY) run_all.py $(RUN_ARGS) --ext lineups

ext-winprob: ## winprob: validate, features, train
	$(PY) run_all.py $(RUN_ARGS) --ext winprob

ext-load: ## load: validate, features, train
	$(PY) run_all.py $(RUN_ARGS) --ext load

ext-betting: ## betting: validate, features, train (depends on lineups)
	$(PY) run_all.py $(RUN_ARGS) --ext betting

all: ## every extension in dependency order on real data (make all DATA=/path/to/DATA_ROOT)
	@test -n "$(DATA)" || (echo "make all needs DATA=/path/to/DATA_ROOT (use make all-fixtures for fixtures)"; exit 2)
	$(PY) run_all.py --data "$(DATA)"$(if $(OUT), --out "$(OUT)",) --all --continue-on-error$(SKIP_ARGS)

all-fixtures: ## every extension in dependency order on fixtures (reports under reports/fixtures unless OUT is set)
	$(PY) run_all.py --fixtures$(if $(OUT), --out "$(OUT)",) --all --continue-on-error$(SKIP_ARGS)

collect-bref: ## print (only print) the collector run order and commands; basketball-reference allows 17 requests a minute
	$(PY) run_all.py --collectors

site: ## copy portfolio/hiring_manager_page.html to SITE_DIR/index.html (default site/)
	mkdir -p "$(SITE_DIR)"
	cp portfolio/hiring_manager_page.html "$(SITE_DIR)/index.html"
	@echo "site: $(SITE_DIR)/index.html"

clean-logs: ## remove logs/run_all and the run_all reports (reports/run_all.md, reports/fixtures/)
	rm -rf logs/run_all reports/run_all.md reports/fixtures

.PHONY: results
results: ## write reports/results.md, one summary table per lane copied from reports/<ext>.md (analytics/results.py)
	$(PY) -m analytics.results --reports reports --out reports/results.md

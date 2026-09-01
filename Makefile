PYTHON ?= python

.PHONY: app init stats sync-telegram backup reindex watch-folder format lint

app:
	streamlit run app.py

init:
	$(PYTHON) -m local_rag.cli init

stats:
	$(PYTHON) -m local_rag.cli stats

sync-telegram:
	$(PYTHON) -m local_rag.cli sync-telegram

backup:
	$(PYTHON) -m local_rag.cli backup

reindex:
	$(PYTHON) -m local_rag.cli reindex

watch-folder:
	$(PYTHON) -m watcher.folder_watcher

format:
	$(PYTHON) -m compileall app.py local_rag

lint:
	$(PYTHON) -m compileall app.py local_rag

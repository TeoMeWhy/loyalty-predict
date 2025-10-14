# Define o ambiente Conda
CONDA_ENV=loyalty-predict

# Define o diretório do ambiente virtual
VENV_DIR=.venv

# Define os diretórios
ENGINEERING_DIR=src/engineering
ANALYTICS_DIR=src/analytics


# Configura o ambiente virtual
.PHONY: setup
setup:
	rm -rf $(VENV_DIR)
	@echo "Criando ambiente virtual..."
	python3 -m venv $(VENV_DIR)
	@echo "Ativando ambiente virtual e instalando dependências..."
	. $(VENV_DIR)/bin/activate && \
	pip install pipreqs && \
 	pipreqs src/ --force --savepath requirements.txt && \
	pip install -r requirements.txt


# Executa os scripts
.PHONY: run
run:
	@echo "Ativando ambiente virtual..."
	. $(VENV_DIR)/bin/activate && \
	cd src/engineering && \
	python get_data.py && \
	cd ../analytics && \
	python pipeline_analytics.py

# Alvo padrão
.PHONY: all
all: setup run
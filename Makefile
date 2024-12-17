# define the name of the virtual environment directory
VENV := .venv

build:
	docker build -t my/csx .

# default target, when make executed without arguments
all: run

$(VENV)/bin/activate: requirements.txt
	pip install virtualenv
	virtualenv $(VENV)
	./$(VENV)/bin/python -m pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements.txt
	./$(VENV)/bin/python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# venv is a shortcut target
venv: $(VENV)/bin/activate

run: venv
	./$(VENV)/bin/uvicorn interface:app 
# --reload

unittest: venv
	./$(VENV)/bin/python test_hda.py

clean:
	# git clean -Xdf
	rm -rf $(VENV)
	find . -type d -name '__pycache__' -exec rm -rf {} +

.PHONY: all venv run clean


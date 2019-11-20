# Cifar 10 API

## Prerequisites

* Python 3
* Pip

## Getting started

### Setting up dev environment
```shell script
python3 -m venv ../cifar10
source ../cifar10/bin/activate
pip install -r requirements.txt
```

### Adding new requirements
```shell script
pip install pip-tools
```
Edit requirements.in
```shell script
pip-compile requirements.in
pip install -r requirements.txt
```

### Training a model
```shell script
python src/data/make_dataset.py
python src/features/build_features.py

```

### Starting the API

TODO: add this section.
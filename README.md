# IncunabulaML

IncunabulaML is an educational project that implements foundational machine learning algorithms in Python.  
The goal is clarity and understanding, not competing with production libraries.

---

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/har5hdeep5harma/IncunabulaML
cd IncunabulaML
``` 

### 2. Setting up a Virtual Environment

It is recommended to use a virtual environment before installing dependencies.

On Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install in editable mode:

```bash
pip install -e .
```
This makes the incunabula package available for import.

### 4. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 5. Running Tests

Tests are written with pytest. Run them from the project root:

```bash
pytest
```

## Project Structure

```bash
IncunabulaML/
│
├── incunabula/
│   ├── __init__.py
│   ├── perceptron.py
│   └── ...
│
├── tests/              
│   ├── test_perceptron.py
│   └── ...
├── setup.py
├── requirements.txt
└── README.md

```
(Files shown are examples; more algorithms and tests will be added over time.)

## License
This project is MIT licensed. See the LICENSE file for details.
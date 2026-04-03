# echo-project

Current project files are mostly placeholders, so the main moving part right now
is the Python environment.

## Recommended setup

Use Python 3.12.x if you want the most predictable TensorFlow + Jupyter behavior.
The current `.venv` uses Python 3.13.5 and `import tensorflow as tf` works, but
the first cold import may feel slow inside notebooks.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name echo-project --display-name "echo-project"
```

## Smoke test

```bash
python -c "import time; t=time.time(); import tensorflow as tf; print(tf.__version__, round(time.time()-t, 3))"
```

In the checked environment, the same import completed in roughly 3 to 4 seconds
from the terminal. If a notebook cell still appears stuck for much longer, restart
the kernel and verify it is attached to the recreated `.venv`.

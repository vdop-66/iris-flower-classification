# iris-flower-classification

Interactive demo and API for classifying Iris flowers with a StandardScaler + KNN pipeline.

## Quick start

Follow these steps to see the interactive site locally:

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# 2) Install the dependencies
pip install -r requirements.txt

# 3) Start the Flask dev server from the project root
flask --app app run --debug
```

Then open http://localhost:5000 in your browser to use the form. You can also call the API directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.3, "sepal_width": 2.7, "petal_length": 4.9, "petal_width": 1.8}'
```

You can still run the original training script for console output and charts:

```bash
python iris_classification.py
```

## Publishing your local changes to GitHub

This workspace repository does not have a Git remote configured, so commits stay local by default. To view your changes on GitHub:

1. Create (or choose) a repository on GitHub.
2. Add it as a remote, for example: `git remote add origin https://github.com/<username>/<repo>.git`.
3. Push the current branch: `git push -u origin work` (replace `work` with your branch name if different).
4. Refresh the repository page on GitHub to confirm the changes are visible.

# Contributing

Contributions are welcome, especially those that improve physical realism, validation, numerical stability, testing, and documentation.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Please add tests for new scientific behavior and document assumptions explicitly.

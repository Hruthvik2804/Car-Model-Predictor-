# Contributing to AutoVision AI

Thanks for your interest in improving AutoVision AI!

## Getting set up

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest        # for the test suite
```

## Before opening a pull request

- Keep changes focused and describe them clearly in the PR.
- Run the lightweight tests: `pytest -q tests/`
- Make sure the app still starts: `streamlit run app.py`
- If you add a vehicle, edit `vehicle_catalog.json` with an accurate
  `manufacturer`, `model`, and `year_range`.

## Adding a new zero-shot vehicle

```json
{ "manufacturer": "Toyota", "model": "Supra", "year_range": "2020-2024" }
```

## Reporting issues

Open an issue with the image (or a description), the mode used
(zero-shot / fine-tuned), and what you expected vs. what happened.

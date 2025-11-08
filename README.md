## Second Brain

A quick reference for getting a local Streamlit environment running with [uv](https://docs.astral.sh/uv/) dependency management.

### Prerequisites

- Python 3.10+ (matches the default supported by uv)
- `uv` installed:
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Restart your shell (or follow the installer prompt) so that `uv` is on your `PATH`.

### Create the virtual environment

```sh
cd /path/to/second-brain
uv venv
source .venv/bin/activate
```

If you prefer to avoid activating the environment manually, you can prepend future commands with `uv run`.

### Install dependencies

- With a `requirements.txt` or `pyproject.toml` in place:
  ```sh
  uv pip install -r requirements.txt
  # or, if using pyproject:
  uv sync
  ```

### Run Streamlit with uv

```sh
uv run streamlit run app.py
```

Replace `app.py` with your Streamlit entrypoint if it lives elsewhere (for example `src/app.py`).

### Useful uv commands

- Update packages: `uv pip install --upgrade <package>`
- Remove unused packages: `uv pip cleanup`
- Export the locked environment (helpful for deployment):
  ```sh
  uv pip compile requirements.txt
  ```

You now have a reproducible Streamlit setup managed entirely through uv.


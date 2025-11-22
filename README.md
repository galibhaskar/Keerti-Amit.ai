## Keerti-Amit.AI

An AI-powered learning assistant application built with Streamlit, featuring practice modes, battle challenges, and document ingestion capabilities.

### Project Structure

```
Keerti-Amit.ai/
├── src/                    # Main source code
│   ├── app.py             # Main Streamlit application
│   ├── config/            # Configuration modules
│   │   ├── settings.py    # App settings and constants
│   │   └── models.py      # LLM model configuration
│   ├── core/              # Core business logic
│   │   ├── database/      # Database operations
│   │   ├── embeddings/    # Embedding models
│   │   └── llm/           # LLM providers and generation
│   ├── services/          # Business services
│   │   ├── ingestion/     # Document ingestion and queue
│   │   └── audio/         # Audio processing (speech/tts)
│   ├── tools/             # LangChain tools
│   ├── utils/             # Utility functions
│   └── ui/                # User interface
│       ├── pages/         # Streamlit pages
│       └── navigation.py  # Page routing
├── app.py                 # Entry point (redirects to src.app)
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

### Prerequisites

- Python 3.13+ (as specified in pyproject.toml)
- `uv` installed:
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Restart your shell (or follow the installer prompt) so that `uv` is on your `PATH`.

### Create the virtual environment

```sh
cd /path/to/keerti-amit.ai
uv venv
source .venv/bin/activate
```

If you prefer to avoid activating the environment manually, you can prepend future commands with `uv run`.

### Install dependencies

```sh
uv sync
```

### Run Streamlit

```sh
uv run streamlit run src/app.py
```

### Configuration

Create a `.env` file using`.env.local` with your API keys:


### Features

- **Data Ingestion**: Upload and process documents (PDF, TXT, images, etc.)
- **Practice Mode**: Interactive flashcards and quizzes
- **Battle Mode**: Voice-based interview challenges
- **Vector Database**: ChromaDB for semantic search and RAG

### Useful uv commands

- Update packages: `uv pip install --upgrade <package>`
- Remove unused packages: `uv pip cleanup`
- Export the locked environment (helpful for deployment):
  ```sh
  uv pip compile pyproject.toml
  ```

You now have a reproducible Streamlit setup managed entirely through uv.
---

## Contributors:
1. Veera Surya Bhaskar Gali (suryagali.se@gmail.com)
2. Sahithi Reddy Mallidi (sahithireddy0299@gmail.com)

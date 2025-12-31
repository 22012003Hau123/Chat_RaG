# RAG Chatbot Demo

> A production-ready RAG system with CI/CD pipeline

A modular implementation of a Retrieval-Augmented Generation (RAG) system for question-answering over custom document collections.

## Overview

This project implements a complete RAG pipeline with the following capabilities:
- Multi-format document ingestion (PDF, TXT, Markdown, HTML, DOCX)
- Semantic search using vector embeddings
- Multiple retrieval strategies (similarity, MMR, hybrid)
- Context-aware response generation
- Performance evaluation framework
- RESTful API and web interface

## Features

- Multi-format document support (PDF, TXT, Markdown, HTML, DOCX)
- Advanced retrieval strategies: similarity search, MMR, and hybrid approaches
- YAML-based configuration system
- Built-in evaluation metrics
- FastAPI backend with RESTful endpoints
- Modular, extensible architecture

## Project Structure

```
rag-chatbot-demo/
├── configs/
│   └── rag.yaml              # System configuration
├── data/                     # Document storage
│   └── vector_index/         # FAISS index and metadata
├── docs/
│   └── CREDENTIALS.md        # Credential management guide
├── envs/                     # Environment credential files (gitignored)
├── notebooks/                # Jupyter notebooks for experimentation
├── src/
│   ├── ingest.py            # Document loading and indexing
│   ├── retriever.py         # Search strategies (top-k, MMR, hybrid)
│   ├── prompt.py            # Prompt templates
│   ├── rag_chain.py         # RAG pipeline orchestration
│   └── eval.py              # Evaluation metrics (planned)
├── .env.example             # Example environment variables
├── .env.dev.example         # Development environment template
├── .env.staging.example     # Staging environment template
├── .env.prod.example        # Production environment template
├── app.py                   # FastAPI web application
├── requirements.txt         # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip or conda for package management

### Installation

Clone the repository:
```bash
git clone https://github.com/squirke1/rag-chatbot-demo.git
cd rag-chatbot-demo
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies (requirements file to be added):
```bash
pip install -r requirements.txt
```

### Setting Up Credentials

**IMPORTANT: Never commit API keys to Git!**

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=your-actual-api-key-here
ENVIRONMENT=dev
DEBUG=True
LOG_LEVEL=DEBUG
```

3. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

See [docs/CREDENTIALS.md](docs/CREDENTIALS.md) for detailed credential management instructions.

**Alternative (temporary):**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Configuration

Environment-specific configuration files live under `configs/`:

- `configs/rag.dev.yaml` – default for local development, verbose logging, localhost binding.
- `configs/rag.prod.yaml` – tuned for production deployments, stricter thresholds, public binding.

The active file is resolved in this order:

1. Explicit `--config /path/to/file.yaml` CLI argument.
2. `RAG_CONFIG_PATH=/absolute/or/relative/path.yaml`
3. `RAG_ENV=dev|prod` (defaults to `dev`).

Examples:

```bash
# Use the production settings
export RAG_ENV=prod

# Point to a completely custom configuration
export RAG_CONFIG_PATH=/opt/rag/configs/custom.yaml
```

Feel free to duplicate one of the provided YAML files if you need additional environments (e.g., `rag.test.yaml`).

## Usage

### Document Ingestion

Place documents in the `data/` directory and run ingestion:
```bash
python src/ingest.py --data-dir data
```

This process:
- Loads documents from the specified directory
- Splits text into chunks with configured size and overlap
- Generates embeddings using the configured model
- Creates a FAISS vector index for similarity search

### Running the Application

Start the FastAPI server:
```bash
python app.py
```

Access the application:
- **Chat Interface**: http://localhost:8000/chat
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

The server runs with auto-reload enabled for development.

## Branching Strategy

Local branches `dev` and `test` are available alongside `main`:

- Develop features on `dev`, promote stabilized changes to `test`, and release from `main`.
- Push the new branches upstream once authenticated:  
  `git push -u origin dev` and `git push -u origin test`
- Protect the branches in GitHub as desired (e.g., required reviews or checks).

And access the web interface at `http://localhost:8000`

### Command Line Interface

Query the RAG system directly:
```bash
python src/rag_chain.py --question "What is RAG?"
```

Use MMR retrieval for diversity:
```bash
python src/rag_chain.py --question "What is RAG?" --method mmr
```

Test retrieval only:
```bash
python src/retriever.py --query "RAG systems" --k 5 --method similarity
```

Show retrieved context:
```bash
python src/rag_chain.py --question "What is RAG?" --show-context
```

## Architecture

```
Documents → Ingest → Vector DB → Retriever → RAG Chain → Response
             ↓          ↓           ↓           ↓
           Split     FAISS     Similarity   LLM + Context
          Embed     Index       Search      Generation
```

## Implementation Status

**Completed:**
- ✅ Project structure and configuration system
- ✅ Document ingestion pipeline (`src/ingest.py`)
  - Multi-format document loading
  - Text chunking with overlap
  - Batch embedding generation
  - FAISS vector index creation
- ✅ Retrieval system (`src/retriever.py`)
  - Similarity search (top-k)
  - MMR (Maximal Marginal Relevance)
  - CLI for testing retrieval
- ✅ Prompt management (`src/prompt.py`)
  - System and user prompt templates
  - Context formatting
  - OpenAI messages format support
- ✅ RAG chain (`src/rag_chain.py`)
  - End-to-end question answering
  - Retrieve → Compose → Generate pipeline
  - CLI for testing Q&A
- ✅ Web application (`app.py`)
  - FastAPI setup with lifespan management
  - CORS middleware
  - Request/response models
  - REST API endpoints (/, /health, /ask)
  - HTML chat interface at /chat
  - Server startup script
- ✅ Credential management
  - Environment-based configuration
  - .env file support with python-dotenv
  - Comprehensive documentation

**Planned:**
- Evaluation framework (`src/eval.py`)
- Advanced features (streaming, conversation history)
- Deployment configuration

## Technical Details

### Document Processing
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Supported formats: PDF, TXT, MD, HTML, DOCX

### Embeddings
- Model: sentence-transformers/all-MiniLM-L6-v2
- Vector dimension: 384
- Batch size: 32

### Vector Store
- Implementation: FAISS (Facebook AI Similarity Search)
- Similarity metric: Cosine similarity
- Storage: Local filesystem

### Retrieval
- Default strategy: Similarity search
- Top-k: 5 documents
- Score threshold: 0.7
- Alternative strategies: MMR, hybrid

### Language Model
- Default provider: OpenAI
- Model: GPT-3.5-turbo
- Temperature: 0.1
- Max tokens: 1000

## Development Approach

This project follows an incremental development process with proper Git Flow and CI/CD:

1. Configuration - Define system parameters
2. Document ingestion - Text processing and embedding generation
3. Retrieval - Vector search implementation
4. Prompt engineering - Context formatting
5. Chain integration - End-to-end pipeline
6. Evaluation - Performance metrics
7. API development - Web interface

## Git Flow Branching Strategy

This project uses a **Git Flow variant** optimized for continuous deployment:

### Branch Structure

```
main (production)
  ↑
test (staging)
  ↑
dev (development)
  ↑
feature/* (feature branches)
```

**Note:** This is a variation of standard Git Flow that uses `dev`/`test`/`main` instead of `develop`/`release`/`main`. The workflow principles remain the same, but branch names are tailored for clarity in CI/CD environments.

### Branches

- **`main`** - Production-ready code, protected branch, requires approval for deployment
- **`test`** - Staging/testing environment, auto-deploys to staging for QA validation
- **`dev`** - Active development branch, all features merge here first, runs CI tests
- **`feature/*`** - Individual feature branches (e.g., `feature/add-streaming`)
- **`detailed-comments`** - Special branch with educational comments (parallel to main)

### Workflow

1. **Create feature branch** from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature
   ```

2. **Develop and commit**:
   ```bash
   git add .
   git commit -m "feat: add your feature"
   git push origin feature/your-feature
   ```

3. **Create Pull Request** to `dev` (triggers CI tests)

4. **Merge to dev** → Auto-runs CI pipeline

5. **Promote to test** for staging deployment:
   ```bash
   git checkout test
   git merge dev
   git push origin test  # Auto-deploys to staging
   ```

6. **Promote to main** for production (requires approval):
   ```bash
   git checkout main
   git merge test
   git push origin main  # Auto-deploys to production
   ```

## CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

Runs automatically on PRs and pushes to `dev`, `test`:

- ✅ **Code Quality**: Syntax checking, linting (flake8, black, isort)
- ✅ **Tests**: Import verification, file structure checks
- ✅ **Security**: Secret detection, .gitignore validation

### Staging Deployment (`.github/workflows/deploy-staging.yml`)

Triggers on push to `test` branch:

- 🚀 Automated deployment to staging environment
- 🧪 Pre-deployment tests
- 🏥 Health checks
- 📢 Deployment notifications

### Production Deployment (`.github/workflows/deploy-production.yml`)

Triggers on push to `main` branch:

- 🔒 Requires manual approval (GitHub Environment protection)
- 🔍 Comprehensive security scans
- ✅ Full test suite
- 🚀 Production deployment
- 📊 Deployment tracking
- 🚨 Failure notifications

### Setting Up GitHub Environments

To enable environment protection (recommended for production):

1. Go to your GitHub repo → **Settings** → **Environments**
2. Create `staging` and `production` environments
3. For `production`:
   - Enable **Required reviewers** (add yourself)
   - Set **Wait timer** (optional, e.g., 5 minutes)
   - Add **Deployment branches** rule (only `main`)

This ensures production deployments require manual approval! 🔐

### Why This Branching Strategy?

This project uses **`dev`/`test`/`main`** instead of the traditional Git Flow **`develop`/`release`/`main`** for several reasons:

1. **Clarity in CI/CD**: Branch names directly correspond to environments (dev, test/staging, production)
2. **Intuitive naming**: `test` clearly indicates "testing/staging", `dev` indicates "development"
3. **Industry adoption**: Many modern teams use similar naming (e.g., GitHub's `main`, GitLab's `staging`)
4. **Simplified workflow**: Eliminates confusion between `release` branches and staging environments
5. **CI/CD integration**: Easier to map branches to deployment targets in GitHub Actions

The **workflow principles** remain identical to standard Git Flow—only the branch names differ. This is a common and accepted variation used by many professional development teams.

## References

- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)
- [RAG Papers and Research](https://arxiv.org/abs/2005.11401)

## Future Enhancements

- Support for more LLM providers (Anthropic, local models)
- Advanced chunking strategies
- Multi-modal document support (images, tables)
- Conversation memory and follow-up questions
- Cloud vector database integration
- Docker containerization
## Server Deployment (Ubuntu/Linux)

### 1. Prerequisites (Install Docker)

If you see `Command 'docker-compose' not found`, run these commands:

```bash
# Update package list
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Install Docker Compose
sudo apt-get install -y docker-compose

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. Deploy App

After installing Docker, navigate to your project folder and run:

```bash
# Build and start in background
docker-compose up -d --build
```

- App will run on port `8000`.
- Logs can be viewed with: `docker-compose logs -f`


## Contact

For questions or feedback, please open an issue on GitHub.

---

*Last Updated: November 1, 2025*  
*Status: Full RAG system complete with web interface*

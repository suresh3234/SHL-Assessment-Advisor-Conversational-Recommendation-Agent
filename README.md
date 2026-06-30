# SHL Assessment Recommendation Agent

A FastAPI-based microservice that acts as an intelligent assistant recommending SHL assessments based on role requirements, skills, and constraints.

---

## How to Run Locally

1. **Install Dependencies**:
   Ensure you have Python 3.11 installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   Copy `.env.example` to `.env` and fill in one of the LLM API keys:
   ```bash
   cp .env.example .env
   ```
   Provide either `GROQ_API_KEY` or `OPENROUTER_API_KEY`.

3. **Start the Application**:
   Run the FastAPI application with Uvicorn:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   The service will be available at `http://localhost:8000`.

4. **Run Tests**:
   Run the test suite using `pytest`:
   ```bash
   python -m pytest
   ```

---

## How to Rebuild the Catalog & Search Index

If the product catalog changes, you can rebuild the pre-computed index files (`bm25.pkl`, `faiss.index`, `ids.json`) by running:

```bash
python scripts/build_index.py
```
This script cleans the raw product catalog, embeds the descriptions, and saves the search index files to `app/retrieval/`.

---

## How to Deploy

The application is prepared for Docker-based deployment (e.g., to Render, Railway, Fly.io, or Hugging Face Spaces).

1. **Build the Docker Image**:
   ```bash
   docker build -t shl-rec-agent .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8000:8000 --env-file .env shl-rec-agent
   ```

### Deployment Configuration (e.g. on Render)
- **Runtime**: Docker
- **Environment Variables**:
  - `PORT`: Automatically set by the platform (defaults to 8000)
  - `GROQ_API_KEY` or `OPENROUTER_API_KEY`: Set your live API key.
- **Health Check Path**: `/health` (resolves instantly without contacting the LLM).

---

## Public Endpoint URL

Once deployed, the live service is available at:
`https://shl-recommendation-agent.onrender.com` (or your platform's assigned URL).
- Health Check: `https://shl-recommendation-agent.onrender.com/health`
- Chat Endpoint: `https://shl-recommendation-agent.onrender.com/chat`

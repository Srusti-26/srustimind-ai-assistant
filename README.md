
# SrustiMind – Local AI Assistant (PyTorch + Flask)

SrustiMind is a beginner-friendly, locally running AI assistant for question answering, text summarization, and creative writing. It uses PyTorch with Hugging Face Transformers and exposes both a terminal interface and a simple Flask web UI. No paid APIs are required; it runs fully offline.&#x20;

Here’s how SrustiMind looks in action:

![SrustiMind UI](screenshot.png)

## Features

* Question answering, concise summarization, and creative content generation (story/poem/essay).&#x20;
* Local, offline model execution with PyTorch; no OpenAI API usage.&#x20;
* Web UI with tabs for Chat, Summarize, Creative, and History; progress bar for model loading; star-based feedback; downloadable conversation history.&#x20;
* In-app feedback collection stored as structured JSON (`feedback_data.json`) for later analysis.

## Architecture

Three-tier design:

* **Frontend:** HTML/CSS/JavaScript (Bootstrap, Font Awesome)
* **Backend:** Flask (Python)
* **AI Engine:** TinyLlama-1.1B via Hugging Face Transformers + PyTorch
  This layout, model choice, and stack are documented in your project summary.&#x20;

### Components in Code

* `main.py`: Loads the TinyLlama chat model, handles prompting, generation, conversation history, and feedback persistence; includes CPU/GPU detection and safe fallbacks.&#x20;
* `app.py`: Flask server with background model loading and JSON endpoints for generation, feedback, history, and health/status.&#x20;
* `templates/index.html`: Bootstrap UI with tabs, progress bar, Markdown rendering (with strikethrough disabled to avoid unwanted formatting), feedback widget, and history viewer.&#x20;

## Tech Stack

* **Language/Runtime:** Python
* **Core Libraries:** PyTorch, Transformers, Flask, tqdm, colorama&#x20;
* **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (via Transformers)&#x20;
* **Frontend:** HTML, Bootstrap 5, Font Awesome, Marked.js (Markdown parsing)&#x20;
* **Data Storage:** Local JSON files for feedback and (optional) saved histories&#x20;

## Project Structure

```
SrustiMind/
├─ main.py                   # CLI assistant (model load, generation, feedback)
├─ app.py                    # Flask web server and API
├─ templates/
│  └─ index.html             # Web UI (Bootstrap + JS)
├─ static/                   # (optional) Static assets if added later
├─ screenshot.png            # Project screenshot
├─ feedback_data.json        # Created at runtime for feedback persistence
├─ README.md                 # This file
└─ LICENSE                   # MIT (as per project summary)
```

Notes: `feedback_data.json` is created/updated at runtime by `main.py` and the Flask API feedback route.&#x20;

## Getting Started

### 1) Set up Python environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch transformers flask tqdm colorama
```

(You can add these to `requirements.txt` if you prefer.)

### 2) Run the CLI (terminal) app

```bash
python main.py
```

The CLI lets you choose between answer, summarize, and creative modes, keeps a local conversation history, and saves feedback to `feedback_data.json`.&#x20;

### 3) Run the web app

```bash
python app.py
```

Then visit:

```
http://localhost:5000
```

The web server loads the model in a background thread; you can query `/model_status` for progress. Once loaded, use the Chat/Summarize/Creative tabs, leave star ratings with optional comments, browse history, download it as JSON, or clear it.&#x20;

## API Endpoints (JSON)

* `GET /model_status` – Background model-loading status (progress/messages).&#x20;
* `POST /generate` – `{ prompt, function_type: "question"|"summary"|"creative" }` → `{ response }`.&#x20;
* `POST /feedback` – Store `{ function_type, prompt, response, rating, comments }` into `feedback_data.json`.&#x20;
* `GET /history` – Return session-scoped conversation history.&#x20;
* `POST /clear_history` – Clear session history.&#x20;
* `GET /download_history` – Download session history JSON.&#x20;
* `GET /health` – Simple health check with timestamp and model status.&#x20;

## Prompting Strategy (Built-in)

The app uses role-appropriate system prompts for each task type (factual Q\&A, summarization, and creative writing), which keeps outputs concise for Q\&A, compressive for summaries, and imaginative for creative tasks.&#x20;

## Testing Summary

Manual tests covered factual Q\&A, summarization, creative generation, and the feedback workflow, with stable performance and correct JSON logging of ratings/comments. The testing period was **June 21–22, 2025**, and the project was marked **Functionally Complete** for user testing and feature expansion.&#x20;

## Roadmap

Planned enhancements include speech-to-text input, multi-language support, LoRA-based fine-tuning using feedback, and a RAG module for document Q\&A.&#x20;

## Troubleshooting

* Slow startup: GPU is used if available; otherwise the model falls back to CPU and loads/generates more slowly.&#x20;
* Markdown quirks: The UI sanitizes strikethrough to avoid unintended `<del>` rendering in model outputs.&#x20;
* Template errors: Ensure `templates/index.html` exists before starting Flask.&#x20;

## Contributing

Issues and pull requests are welcome. Please discuss major changes in an issue first.

## License

MIT License (as declared in the project summary).&#x20;

## Acknowledgments

Built with TinyLlama, Transformers, PyTorch, and Flask.&#x20;



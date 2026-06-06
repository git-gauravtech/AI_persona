# Gaurav Saklani AI Persona

An end-to-end AI persona built for the **Scaler AI Engineer Intern Screening Assignment**.

The system allows recruiters to interact with Gaurav Saklani’s AI representative through both:

* A public chat interface
* A phone-call based voice agent

The persona can answer questions about Gaurav’s resume, education, projects, GitHub repositories, achievements, and role fit. It is grounded using RAG over Gaurav’s resume, GitHub README files, commits, and contribution data. It can also check real calendar availability and book or cancel interviews using Cal.com.

---

## Live Links

| Item                     | Link                                          |
| ------------------------ | --------------------------------------------- |
| Public Chat URL          | 'https://scaler-ai-persona-phi.vercel.app/'                   |
| Voice Agent Phone Number | `+1 669-268-0328`                                |

---

## Features

### Chat Interface

* Answers questions about Gaurav’s background, education, skills, projects, GitHub repositories, and achievements.
* Uses RAG over real resume and GitHub data.
* Supports follow-up questions using conversation history.
* Can schedule and cancel interviews directly from chat.
* Handles adversarial prompts and refuses to invent unverified information.

### Voice Agent

* Public phone number powered by Vapi.
* Introduces itself as Gaurav Saklani’s AI representative.
* Handles natural conversation, interruptions, and off-script questions.
* Can answer role-fit, project, and resume questions.
* Can ask availability, fetch real Cal.com slots, confirm booking, and cancel bookings.
* Uses stricter booking-stage validation to avoid accidental booking/cancellation errors.

### Booking System

* Uses Cal.com v2 APIs for real availability and booking.
* Shows real available slots.
* Collects attendee name and email.
* Confirms before creating calendar booking.
* Supports cancellation with reason.
* Keeps booking state per session.

### Reliability

* Uses a shared `llm_client.py` with 4 Groq API key fallback.
* Uses primary and fallback Groq models.
* Prevents raw Groq rate-limit errors from being shown to users.
* Uses UptimeRobot-compatible `/health` endpoint with both `GET` and `HEAD`.

---

## Architecture

```mermaid
flowchart TD
    A[Recruiter / User] --> B1[Public Chat UI<br/>Vercel]
    A --> B2[Phone Call<br/>Vapi]

    B1 --> C[FastAPI Backend<br/>Render]
    B2 --> C

    C --> D[Intent Detection<br/>intent.py]
    C --> E[Active Booking Router<br/>chat_api.py]
    C --> F[RAG Retrieval<br/>retrieve.py]
    C --> G[Booking Flow<br/>booking.py]

    F --> H[Pinecone Vector DB]
    H --> F

    F --> I[Indexed Data Corpus]
    I --> I1[Resume]
    I --> I2[GitHub READMEs]
    I --> I3[Commit History]
    I --> I4[PR / Contribution Notes]

    G --> J[Cal.com API]
    J --> G

    C --> K[LLM Client<br/>llm_client.py]
    D --> K
    E --> K
    F --> K
    G --> K

    K --> L1[Groq Key 1]
    K --> L2[Groq Key 2]
    K --> L3[Groq Key 3]
    K --> L4[Groq Key 4]

    L1 --> M[Groq Models<br/>70B Primary / 8B Fallback]
    L2 --> M
    L3 --> M
    L4 --> M

    M --> C
    C --> B1
    C --> B2
```

---

## High-Level Flow

### Chat Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Vercel Chat UI
    participant Backend as FastAPI Backend
    participant Intent as Intent Router
    participant RAG as Pinecone RAG
    participant LLM as Groq via llm_client
    participant Cal as Cal.com

    User->>Frontend: Ask question or request booking
    Frontend->>Backend: POST /chat
    Backend->>Intent: Detect intent

    alt Booking request
        Backend->>Cal: Fetch availability / create booking / cancel booking
        Cal-->>Backend: Slot or booking result
        Backend-->>Frontend: Booking response
    else Resume / Project question
        Backend->>RAG: Retrieve relevant chunks
        RAG-->>Backend: Context
        Backend->>LLM: Generate grounded answer
        LLM-->>Backend: Answer
        Backend-->>Frontend: Response
    end

    Frontend-->>User: Display answer
```

### Voice Flow

```mermaid
sequenceDiagram
    participant Caller
    participant Vapi
    participant Backend as FastAPI Backend
    participant Router as Intent + Booking Router
    participant RAG as Pinecone RAG
    participant Cal as Cal.com
    participant LLM as Groq via llm_client

    Caller->>Vapi: Speaks on phone
    Vapi->>Vapi: Speech-to-text using Deepgram
    Vapi->>Backend: POST /vapi/chat/completions
    Backend->>Router: Decide route

    alt Active booking
        Router->>Cal: Check slots / book / cancel
        Cal-->>Router: Calendar result
        Router-->>Backend: Booking reply
    else Background or project question
        Router->>RAG: Retrieve context
        RAG-->>Router: Relevant chunks
        Router->>LLM: Generate answer
        LLM-->>Router: Voice-friendly reply
    end

    Backend-->>Vapi: OpenAI-compatible response
    Vapi->>Caller: Text-to-speech response
```

---

## Tech Stack

### Frontend

* HTML
* CSS
* Vanilla JavaScript
* Vercel deployment
* Runtime API config using generated `config.js`

### Backend

* Python
* FastAPI
* Uvicorn
* Render deployment
* Pydantic
* HTTPX

### AI / LLM

* Groq API
* `llama-3.3-70b-versatile` for high-quality main answers
* `llama-3.1-8b-instant` for intent routing, extraction, and fallback
* Multi-key fallback through `llm_client.py`

### RAG

* Pinecone vector database
* FastEmbed embeddings
* Resume + GitHub README + commits + PR/contribution data
* Smart retrieval for chat and voice contexts

### Voice

* Vapi
* Deepgram transcription
* Vapi voice
* Phone number attached to assistant

### Calendar

* Cal.com v2 API
* Real slot fetching
* Real booking confirmation
* Cancellation with reason

---

## Repository Structure

```text
scaler-ai-persona/
├── backend/
│   ├── chat_api.py
│   ├── booking.py
│   ├── intent.py
│   ├── retrieve.py
│   ├── llm_client.py
│   ├── parse_data.py
│   ├── embed_and_upsert.py
│   ├── requirements.txt
│   ├── .python-version
│   └── data/
│       ├── resume/
│       ├── repos/
│       └── processed/
│
├── frontend/
│   ├── index.html
│   ├── generate-config.js
│   ├── package.json
│   └── vercel.json
│
├── README.md
└── .gitignore
```

Note: the `data/` folder is excluded from GitHub because it contains local processed data. Pinecone stores the deployed vector index.

---

## Environment Variables

### Backend `.env`

```env
# Groq multi-key fallback
GROQ_API_KEY_1=your_first_groq_key
GROQ_API_KEY_2=your_second_groq_key
GROQ_API_KEY_3=your_third_groq_key
GROQ_API_KEY_4=your_fourth_groq_key

GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
GROQ_ROUTER_MODEL=llama-3.1-8b-instant
CONTACT_EXTRACTION_MODEL=llama-3.1-8b-instant
INTENT_MODEL=llama-3.1-8b-instant

# Pinecone
PINECONE_API_KEY=your_pinecone_key

# Cal.com
CAL_API_KEY=your_cal_api_key
CAL_EVENT_TYPE_ID=your_event_type_id
CAL_USERNAME=gaurav-saklani
```

### Frontend Vercel Environment Variable

```env
API_URL=https://scaler-ai-persona-o9om.onrender.com/chat
```

---

## Local Setup

### 1. Clone the repo

```bash
git clone ADD_YOUR_REPO_URL_HERE
cd scaler-ai-persona
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add environment variables

Create:

```text
backend/.env
```

Add the variables listed above.

### 4. Run backend locally

```bash
uvicorn chat_api:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl http://localhost:8000/health
```

---

## Data Pipeline

### 1. Prepare data

The data folder contains:

```text
data/
├── resume/
│   └── resume.txt
├── repos/
│   ├── personal/
│   └── contributed/
└── processed/
```

Each repository folder contains selected README, commits, and contribution data.

### 2. Parse data

```bash
python parse_data.py
```

This creates:

```text
data/processed/parsed_chunks.json
```

### 3. Embed and upsert to Pinecone

```bash
python embed_and_upsert.py
```

This creates or updates the Pinecone index:

```text
Index: ai-persona-local
Namespace: gaurav-ai-persona
```

---

## API Endpoints

### Health

```http
GET /health
HEAD /health
```

Used by UptimeRobot and deployment checks.

### Chat

```http
POST /chat
```

Request:

```json
{
  "message": "Tell me about Gaurav's AI projects",
  "session_id": "test-session",
  "history": []
}
```

### Voice

```http
POST /voice
```

Request:

```json
{
  "message": "Why is Gaurav fit for this role?",
  "session_id": "voice-test"
}
```

### Vapi Custom LLM

```http
POST /vapi/chat/completions
POST /vapi
POST /chat/completions
```

These endpoints support Vapi’s Custom LLM / OpenAI-compatible format.

---

## Deployment

### Backend on Render

Render settings:

```text
Root Directory: backend
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: uvicorn chat_api:app --host 0.0.0.0 --port $PORT
```

Python version:

```text
3.11.9
```

Health URL:

```text
https://scaler-ai-persona-o9om.onrender.com/health
```

### Frontend on Vercel

Vercel settings:

```text
Root Directory: frontend
Framework Preset: Other
Build Command: npm run build
Output Directory: .
```

The frontend uses `generate-config.js` to create `config.js` during build:

```js
window.APP_CONFIG = {
  API_URL: "https://scaler-ai-persona-o9om.onrender.com/chat"
};
```

---

## Vapi Configuration

### Model Settings

```text
Provider: Custom LLM
Model: gaurav-ai-persona
Custom LLM URL: https://scaler-ai-persona-o9om.onrender.com/vapi
Temperature: 0.3
Max Tokens: 220
```

### First Message

```text
Hi, I’m Gaurav Saklani’s AI representative. I can answer questions about his background, projects, GitHub work, and availability, and I can help schedule an interview.
```

### Transcriber

```text
Provider: Deepgram
Model: Flux General English
Language: English
Smart Endpointing: Off
End of Turn Timeout: around 1600ms
```

### Phone Number

```text
+16692680328
```

---

## Reliability Improvements

### Multi-Key Groq Fallback

All LLM calls go through `llm_client.py`.

Fallback order:

```text
GROQ_API_KEY_1 + primary model
GROQ_API_KEY_2 + primary model
GROQ_API_KEY_3 + primary model
GROQ_API_KEY_4 + primary model
GROQ_API_KEY_1 + fallback model
GROQ_API_KEY_2 + fallback model
GROQ_API_KEY_3 + fallback model
GROQ_API_KEY_4 + fallback model
```

This prevents one exhausted key from breaking the full application.

### Booking State Protection

The booking flow uses state-based handling.

Critical booking stages bypass general routing:

```text
collecting_info
awaiting_confirmation
cancel_reason
cancel_confirming
```

This prevents a cancellation reason or attendee email from being misclassified as a normal chat message.

### Active Booking Pause

During booking, the user can ask a normal question such as:

```text
Tell me about Vocalis first.
```

The system can pause booking, answer using RAG, and then continue the booking flow.

---

## Cost Breakdown

Approximate cost depends on traffic and free-tier limits.

| Component   | Cost Notes                                                                 |
| ----------- | -------------------------------------------------------------------------- |
| Render      | Free tier used for backend hosting                                         |
| Vercel      | Free tier used for frontend hosting                                        |
| Pinecone    | Serverless/free usage for small vector index                               |
| Groq        | Free/on-demand tier; multi-key fallback used to reduce rate-limit failures |
| Vapi        | Charged per voice call minute                                              |
| Cal.com     | Used for real calendar booking                                             |
| UptimeRobot | Free monitor used to keep backend warm                                     |

Approximate per interaction:

```text
Chat session: low cost, mostly Groq tokens + Pinecone query
Voice call: Vapi call minutes + Groq tokens + Pinecone queries
Booking: Cal.com API calls + Groq parsing calls
```

---

## Known Failure Modes and Fixes

### 1. Groq rate limit caused raw 429 error

Root cause: all Groq calls used one API key.

Fix: added `llm_client.py` with 4-key fallback and fallback model.

### 2. Voice transcription misunderstood booking slot choices

Root cause: speech-to-text sometimes produced unclear slot/date phrases.

Fix: stricter slot validation. After slots are shown, the user must choose a shown option or ask for more slots.

### 3. Cancellation reason was mistaken as stopping the booking flow

Root cause: active booking router intercepted messages during cancellation stages.

Fix: strict booking stages now go directly to `booking.py`, bypassing general route classification.

---

---

## Author

**Gaurav Saklani**
B.Tech CSE with AI/ML Specialization
Graphic Era Hill University, Dehradun

GitHub: https://github.com/git-gauravtech
LinkedIn: https://linkedin.com/in/gaurav-saklani-06a17a300

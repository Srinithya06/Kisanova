# Kisanova

Kisanova is an AI-powered government-scheme discovery assistant that helps Indian farmers find potentially relevant central and state-specific schemes through natural-language questions.

## 🌾 Problem

Farmers often need to search across multiple government portals and departments to find support that may apply to their situation. Scheme names, benefits, required documents, and eligibility conditions can be difficult to compare, especially when state and central programmes use different processes.

Kisanova addresses four connected problems:

- Farmers may not know which government schemes could be relevant to them.
- Scheme information is scattered across portals and departments.
- Eligibility conditions are often difficult to understand from official programme descriptions alone.
- Farmers need a simple way to discover both central and state-specific support.

## 💡 Solution

Farmers describe their situation conversationally, for example:

> I'm a farmer from Karnataka with 3 acres and I grow paddy. What schemes may be relevant to me?

Kisanova then:

1. Uses AI to extract structured profile information such as state, land area, crops, farmer type, and income when those details are actually provided.
2. Uses a deterministic Python eligibility engine to match that profile against the scheme database.
3. Sends only the matched scheme information to the AI explanation layer.
4. Presents the results in a dashboard with counts, recommendations, filters, benefits, eligibility text, documents, and official links.

The LLM does not make the final eligibility decision. Matching is performed by Python rules; the AI is used for language understanding and explanation.

## ✨ Key Features

- Natural-language farmer profile extraction
- Deterministic eligibility matching using state, crop, and land-area parameters
- Central and state-specific scheme discovery
- Scheme recommendations and match counts
- Benefit, eligibility, and document explanations
- State-specific and central-scheme filtering
- Scheme explorer search and category filters
- AI-generated explanations grounded in matched scheme data
- Clarification when important profile information is missing or ambiguous
- Responsive web dashboard built with vanilla HTML, CSS, and JavaScript
- Official scheme and application links where available
- Local fallback extraction and explanation when Groq is unavailable

## 🧠 How It Works

```text
Farmer Message
     ↓
LLM Profile Extraction
     ↓
Structured Farmer Profile
     ↓
Deterministic Python Eligibility Engine
     ↓
Matched Government Schemes
     ↓
Grounded AI Explanation
     ↓
Kisanova Dashboard
```

The system separates language understanding from eligibility matching. Groq extracts facts from the farmer's message, including only information that is present or confidently resolved. The Python engine then checks the profile against `schemes.json` using deterministic state, crop, and acreage rules.

This design prevents the LLM from inventing an eligibility decision. A match means that the available profile parameters align with the stored scheme parameters. It does not replace the government's own verification process or additional scheme-specific conditions.

## 🏗️ Architecture

### Frontend

`templates/index.html` contains the responsive dashboard, including the natural-language query form, loading and clarification states, result rendering, recommendations, state/central badges, scheme search, category filters, and official links. It uses HTML, CSS, and existing browser JavaScript without a frontend framework or build step.

### Backend

`main.py` runs the Flask application. It serves the dashboard and exposes REST endpoints for the assistant flow, scheme data, health checks, sessions, eligibility checks, field guesses, and valid values.

### AI and profile extraction

`kisanova_engine.py` provides the provider abstraction, Groq integration, structured farmer-profile extraction, conservative explanation prompting, and local fallback behavior. The Groq model defaults to `openai/gpt-oss-20b` and can be configured with `GROQ_MODEL`.

### Eligibility engine

`chatbot.py` contains the deterministic scheme loader, state and crop matching, acreage checks, valid-value lists, and the original interactive chatbot logic.

### Scheme database

`schemes.json` stores the current scheme catalogue, including matching parameters, benefits, eligibility descriptions, documents, categories, status, deadlines, and official URLs.

### API layer

The primary dashboard endpoint is `POST /api/kisanova/assist`. Other available endpoints include:

- `GET /health`
- `GET /api/schemes`
- `GET /api/schemes/<scheme_id>`
- `POST /api/check-eligibility`
- `POST /api/session/create`
- `POST /api/chat`
- `POST /api/guess/<field>`
- `GET /api/valid-values/<field>`

## 📊 Scheme Coverage

The current database contains **40 schemes**: central government entries and state-specific programmes. It is a curated project dataset, not a claim to include every government scheme in India.

State-specific schemes currently represented in `schemes.json` cover:

**Andhra Pradesh, Assam, Bihar, Gujarat, Haryana, Himachal Pradesh, Jammu & Kashmir, Karnataka, Kerala, Madhya Pradesh, Maharashtra, Meghalaya, Nagaland, Odisha, Punjab, Rajasthan, Sikkim, Tamil Nadu, Telangana, Tripura, Uttar Pradesh, Uttarakhand, and West Bengal.**

Each scheme can include:

- Eligibility parameters
- State and district scope
- Crop scope
- Minimum and maximum land-area constraints where applicable
- Benefit descriptions
- Required documents when available
- Official URLs
- Category, deadline, and active status fields

State coverage and scheme details should be verified against the relevant official government portal before applying.

## 🖥️ User Experience

1. The farmer enters a natural-language question describing their farming situation.
2. Kisanova identifies the available farmer profile fields and asks for clarification when required details are missing or ambiguous.
3. The dashboard displays the profile and the number of matching schemes.
4. Results separate state-specific and central schemes and highlight top recommendations.
5. The AI insight panel explains why the returned schemes may be relevant.
6. The scheme explorer lets users search by name and filter by state/central scope or category.
7. Each scheme card shows the stored benefit, eligibility text, documents, and official portal link.

## 🔐 Eligibility & AI Grounding

Kisanova uses the LLM for language understanding and explanation, not as the final eligibility authority.

- Python performs deterministic matching against the stored scheme parameters.
- AI explanations are grounded in the matched scheme data supplied to the explanation layer.
- Missing farmer attributes are not treated as facts. The system avoids assuming land ownership, farmer category, tenancy, caste, income, age, or similar details that were not provided.
- Additional conditions in a scheme's stored eligibility description are surfaced as conditions to verify, rather than being silently converted into a guaranteed result.
- A scheme appearing as a profile match does not guarantee approval, payment, insurance coverage, or any other government benefit.
- Final eligibility, document acceptance, enrolment, and approval are determined by the responsible government department and its official portal.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| AI | Groq LLM |
| Eligibility Engine | Python |
| Data | JSON |
| API | REST |

## 📁 Project Structure

```text
Kisanova/
├── main.py                  # Flask application and REST endpoints
├── chatbot.py               # Scheme loading and deterministic matching
├── kisanova_engine.py       # Groq extraction, explanations, and fallback provider
├── schemes.json             # Scheme catalogue and matching metadata
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python runtime version
├── templates/
│   └── index.html            # Kisanova web dashboard
├── test_kisanova.py         # Automated unittest coverage
├── verify_live_groq.py      # Optional live Groq verification
├── verify_live_gemini.py    # Provider verification utility
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/Srinithya06/Kisanova.git
cd Kisanova
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Groq

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

The API key is used for profile extraction and explanation generation. If Groq is unavailable or the key is missing, Kisanova uses its local fallback provider.

### 5. Start the application

```bash
python main.py
```

The Flask app uses port `8000` by default. Open the URL printed by the application, typically `http://localhost:8000/`.

## 🧪 Testing

Run the repository's unittest suite:

```bash
python -m unittest test_kisanova.py
```

For an optional live Groq integration check:

```bash
python verify_live_groq.py
```

## ⚖️ Disclaimer

Kisanova provides scheme discovery and potentially relevant information based on the supplied profile and the current local dataset. It is not a government authority. Always verify current eligibility, documents, deadlines, enrolment, and approval requirements on the official scheme portal before applying.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
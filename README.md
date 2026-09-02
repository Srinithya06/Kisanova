# 🌾 Kisanova

### AI-Powered Government Scheme Discovery for Farmers

Kisanova is an AI-powered platform that helps farmers discover government schemes they may be eligible for based on their personal and farming details.

Instead of searching through multiple government websites and complicated eligibility documents, farmers can simply describe their situation in natural language.

For example:

> "I'm a farmer from Telangana with 3 acres of land and I grow paddy. What schemes can I apply for?"

Kisanova extracts the farmer's details, checks them against the available scheme data using deterministic eligibility rules, and provides clear explanations about relevant schemes.

---

## 🚀 Features

- 🌱 Natural-language farmer queries
- 🤖 AI-powered farmer profile extraction using Groq
- ✅ Deterministic scheme eligibility matching
- 🇮🇳 Central and state-specific government schemes
- 📋 Scheme benefits and eligibility information
- 📄 Required document information
- 🔗 Official application links
- 💬 AI-generated explanations of matching schemes
- ❓ Clarification when required farmer information is missing
- 🌐 Simple responsive web interface
- 🧪 Automated test coverage

---

## 🏗️ Architecture

```text
Farmer Query
     │
     ▼
┌─────────────────────┐
│   Kisanova Web UI   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Flask Backend    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Groq LLM          │
│ Profile Extraction  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Deterministic       │
│ Eligibility Engine  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    schemes.json     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Matching Schemes    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Groq Explanation    │
└─────────────────────┘


Important Design Principle

The LLM does not decide whether a farmer is eligible.

The system follows a two-stage approach:

AI extraction — Groq extracts structured information from the farmer's natural-language query.
Deterministic matching — Python checks the farmer's profile against the scheme eligibility rules.

The LLM is then used only to explain the matched schemes.

🛠️ Technology Stack
Python
Flask
Groq API
HTML / CSS / JavaScript
JSON
Git / GitHub
📁 Project Structure
Kisanova/
│
├── chatbot.py
├── kisanova_engine.py
├── main.py
├── schemes.json
├── templates/
│   └── index.html
├── test_kisanova.py
├── verify_live_groq.py
├── requirements.txt
├── runtime.txt
├── .gitignore
├── LICENSE
└── README.md
⚙️ Setup
1. Clone the repository
git clone https://github.com/Harshith0906/Kisanova.git
cd Kisanova
2. Create a virtual environment
python -m venv venv

Activate it on Windows:

venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure Groq

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

Never commit your .env file or API key to GitHub.

5. Run Kisanova
python main.py

Then open the local URL shown by Flask in your browser.

🧪 Testing

Run:

pytest test_kisanova.py -v

The project also includes a live Groq verification script:

python verify_live_groq.py
🌾 Example Query
I'm a farmer from Telangana with 3 acres of land and I grow paddy.
What government schemes can I apply for?

Kisanova identifies the relevant farmer information, matches eligible schemes, and presents their benefits, eligibility requirements, documents, and application information.

🎯 Project Goal

Kisanova aims to make government agricultural schemes easier to discover and understand, especially for farmers who may find government portals and eligibility documents difficult to navigate.

📜 License

This project is licensed under the MIT License.


### Step 3 — Save it

Press:

**Ctrl + S**

That's it for now.

**Don't commit or push yet.** Tell me when you've saved it, and I'll give you the next step.
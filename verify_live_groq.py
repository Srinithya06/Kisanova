import os
import sys
import json
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from kisanova_engine import (
    GroqProvider,
    find_kisanova_schemes,
    GROQ_AVAILABLE
)

def run_live_groq_verification():
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY")

    provider = GroqProvider()
    test_message = "I am a farmer from Telangana with 3 acres growing paddy."

    extract_res = provider.extract_farmer_profile(test_message)
    extraction_engine = extract_res.get("engine_used", "fallback")

    profile = extract_res.get("farmer_profile", {})
    matched_schemes = find_kisanova_schemes(profile)

    explanation_text, explanation_engine = provider.generate_explanation(profile, matched_schemes)

    is_live_success = (extraction_engine == "groq" and explanation_engine == "groq")
    fallback_used = (extraction_engine == "fallback" or explanation_engine == "fallback")

    print("=" * 60)
    print("KISANOVA LIVE GROQ VERIFICATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"HTTP/API Result: {'SUCCESS (200 OK)' if is_live_success else 'FAILED / REJECTED (Missing or Invalid GROQ_API_KEY)'}")
    print(f"Groq Profile Extraction: {'YES' if extraction_engine == 'groq' else 'NO'}")
    print(f"Groq Explanation Generation: {'YES' if explanation_engine == 'groq' else 'NO'}")
    print(f"Fallback Used: {'YES' if fallback_used else 'NO'}")
    print(f"Groq Integration Verified: {'YES' if is_live_success else 'NO (Ran on Fallback Engine)'}")
    print("=" * 60)

if __name__ == "__main__":
    run_live_groq_verification()

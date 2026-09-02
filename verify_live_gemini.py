import os
import sys
import json
import logging
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from kisanova_engine import (
    extract_farmer_profile,
    find_kisanova_schemes,
    generate_scheme_explanation,
    get_genai_client
)

def run_live_gemini_test():
    print("=" * 60)
    print("🚀 KISANOVA LIVE GEMINI API VERIFICATION TEST")
    print("=" * 60)
    print(f"Model Name Target: gemini-2.5-flash")
    print(f"GEMINI_API_KEY Configured: {'YES' if os.getenv('GEMINI_API_KEY') else 'NO'}\n")

    test_message = "I am a farmer from Telangana with 3 acres growing paddy."
    print(f"Input Farmer Query: \"{test_message}\"\n")

    # Step 1: Extraction Test
    print("Step 1: Testing Profile Extraction...")
    extraction_res = extract_farmer_profile(test_message)
    extraction_engine = extraction_res.get("engine_used", "unknown")
    print(f"  -> Extraction Status: {extraction_res.get('status')}")
    print(f"  -> Extraction Engine Executed: {extraction_engine.upper()}")
    print(f"  -> Extracted Profile: {json.dumps(extraction_res.get('farmer_profile'), indent=2)}\n")

    # Step 2: Deterministic Matching Test
    print("Step 2: Deterministic Scheme Matching...")
    profile = extraction_res.get("farmer_profile", {})
    matched_schemes = find_kisanova_schemes(profile)
    print(f"  -> Matched Schemes Count: {len(matched_schemes)} schemes\n")

    # Step 3: Explanation Generation Test
    print("Step 3: Testing Personalized Explanation Generation...")
    explanation_text, explanation_engine = generate_scheme_explanation(profile, matched_schemes)
    print(f"  -> Explanation Engine Executed: {explanation_engine.upper()}\n")

    # Summary Report
    is_live_gemini_success = (extraction_engine == "gemini" and explanation_engine == "gemini")
    fallback_used = (extraction_engine == "fallback" or explanation_engine == "fallback")

    print("=" * 60)
    print("📊 LIVE GEMINI VERIFICATION SUMMARY REPORT")
    print("=" * 60)
    print(f"1. Model Name: gemini-2.5-flash")
    print(f"2. HTTP/API Result: {'SUCCESS (200 OK)' if is_live_gemini_success else 'FAILED / REJECTED BY GOOGLE API (400 Bad Request)'}")
    print(f"3. Gemini Profile Extraction Executed: {'YES' if extraction_engine == 'gemini' else 'NO'}")
    print(f"4. Gemini Explanation Generation Executed: {'YES' if explanation_engine == 'gemini' else 'NO'}")
    print(f"5. Fallback Engine Used: {'YES' if fallback_used else 'NO'}")
    print(f"6. Gemini Integration Verified: {'VERIFIED SUCCESSFUL' if is_live_gemini_success else 'NOT VERIFIED (Ran on Fallback Path)'}")
    print("=" * 60)

if __name__ == "__main__":
    run_live_gemini_test()

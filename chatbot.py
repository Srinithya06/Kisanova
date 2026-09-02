import json
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import random

# ==================== ML WORD GUESSER ====================

class MLWordGuesser:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        self.nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.words = []
        self.is_trained = False
    
    def train(self, correct_words):
        self.words = correct_words
        X = self.vectorizer.fit_transform(correct_words)
        self.nn_model.fit(X)
        self.is_trained = True
        return self
    
    def guess(self, user_input):
        if not self.is_trained or not user_input:
            return None
        user_vector = self.vectorizer.transform([user_input.lower().strip()])
        distances, indices = self.nn_model.kneighbors(user_vector)
        return {
            'word': self.words[indices[0][0]],
            'confidence': float(1 - distances[0][0])
        }


# ==================== CORRECT WORDS ====================

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

VALID_CROPS = [
    "wheat", "rice", "cotton", "sugarcane", "dairy",
    "livestock", "vegetables", "fruits", "millets", "pulses"
]

FARMER_TYPES = ["individual", "shg", "fpo", "other"]

YES_NO = ["yes", "no", "y", "n"]


# ==================== TRAIN ML MODELS ====================

state_guesser = MLWordGuesser().train(INDIAN_STATES)
crop_guesser = MLWordGuesser().train(VALID_CROPS)
type_guesser = MLWordGuesser().train(FARMER_TYPES)


# ==================== HELPER FUNCTIONS ====================

def extract_number(text):
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return float(numbers[0])
    return None


# ==================== CORE LOGIC ====================

def load_schemes(path="schemes.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("schemes", {})


def normalize_to_list(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    if value is None:
        return []
    return [str(value).strip()]


def state_matches(scheme_state, farmer_state):
    if scheme_state == "ALL":
        return True
    if isinstance(scheme_state, list):
        return farmer_state.title() in [s.title() for s in scheme_state]
    return farmer_state.title() == str(scheme_state).title()


def crops_match(scheme_crops, farmer_crops):
    scheme_crops = [c.lower() for c in normalize_to_list(scheme_crops)]
    farmer_crops = [c.lower() for c in farmer_crops]
    if "all" in scheme_crops:
        return True
    return any(fc in scheme_crops for fc in farmer_crops)


def acres_match(min_acres, max_acres, farmer_acres, scheme_crops, farmer_crops):
    try:
        min_a = float(min_acres)
    except:
        min_a = 0.0
    try:
        max_a = float(max_acres)
    except:
        max_a = float("inf")

    if max_a == 0.0:
        lcrops = [c.lower() for c in normalize_to_list(scheme_crops)]
        if any(x in lcrops for x in ("livestock", "dairy", "all")):
            return any(fc in lcrops for fc in [c.lower() for c in farmer_crops])
        return farmer_acres == 0

    return (farmer_acres >= min_a) and (farmer_acres <= max_a)


def find_eligible_schemes(schemes, farmer):
    """Deterministically matches a farmer profile against the available government schemes."""
    matches = []
    farmer_state = farmer.get("state", "").strip() if farmer.get("state") else ""
    farmer_crops = [c.strip() for c in farmer.get("crops", []) if c.strip()]
    farmer_acres = float(farmer.get("acres", 0)) if farmer.get("acres") is not None else 0.0

    scheme_iterable = schemes.values() if isinstance(schemes, dict) else schemes

    for scheme in scheme_iterable:
        params = scheme.get("params", {})
        scheme_state = params.get("state", "ALL")
        scheme_crops = params.get("crops", ["ALL"]) or ["ALL"]
        min_acres = params.get("min_acres", 0)
        max_acres = params.get("max_acres", float("inf"))

        if not state_matches(scheme_state, farmer_state):
            continue
        if not crops_match(scheme_crops, farmer_crops):
            continue
        if not acres_match(min_acres, max_acres, farmer_acres, scheme_crops, farmer_crops):
            continue

        matches.append(scheme)

    return matches


def rank_and_display_schemes(schemes_list, farmer):
    if not schemes_list:
        print("\nNo matching schemes found.")
        return

    # Calculate relevance score for each scheme
    scored_schemes = []
    for scheme in schemes_list:
        score = 0
        params = scheme.get("params", {})
        
        # State match bonus
        if params.get("state") != "ALL":
            if isinstance(params.get("state"), list):
                if farmer["state"] in params["state"]:
                    score += 3  # Specific state match
            else:
                if farmer["state"] == params.get("state"):
                    score += 3  # Specific state match
        
        # Crop match bonus
        scheme_crops = normalize_to_list(params.get("crops", []))
        farmer_crops = [c.lower() for c in farmer["crops"]]
        if "all" not in [c.lower() for c in scheme_crops]:
            # Check if any of farmer's crops match scheme's specific crops
            matching_crops = [c for c in farmer_crops if c in [sc.lower() for sc in scheme_crops]]
            if matching_crops:
                score += 3  # Specific crop match
        
        # Land requirement match (if farmer has land)
        if farmer["acres"] > 0:
            min_acres = params.get("min_acres", 0)
            max_acres = params.get("max_acres", float("inf"))
            if min_acres > 0 or max_acres != float("inf"):
                score += 1  # Has land requirements
        
        # Add score to scheme
        scored_schemes.append((score, scheme))
    
    # Sort by score (highest first)
    scored_schemes.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\nFound {len(schemes_list)} matching scheme(s). Showing most relevant first:\n")
    
    # Display first 4 schemes with full details
    for i, (score, scheme) in enumerate(scored_schemes[:4], 1):
        print(f"{i}. {scheme.get('name')} (ID: {scheme.get('id')})")
        
        # Add relevance indicator
        if score >= 3:
            print("   ⭐ Highly relevant for you")
        
        print(f"   Benefit: {scheme.get('benefit')}")
        docs = scheme.get('docs', [])
        if docs:
            print(f"   Documents: {', '.join(docs)}")
        if scheme.get('url'):
            print(f"   Apply: {scheme.get('url')}")
        print("")
    
    # Display remaining schemes (only names)
    if len(scored_schemes) > 4:
        print("Other schemes you may be eligible for:")
        for i, (score, scheme) in enumerate(scored_schemes[4:], 5):
            relevance = "⭐" if score >= 3 else ""
            print(f"   {i}. {scheme.get('name')} {relevance}")
        print("")




# ==================== CHATBOT ====================

def run():
    schemes = load_schemes()

    print(" FARMER SCHEME ASSISTANT ")
    
    # Simple greeting handler
    print("\n👋 Hi! I'm your farming scheme assistant.")
    print("I can help you find government schemes based on your farming details.")
    
    while True:
        initial = input("\nReady to start? (yes/no/hello): ").strip().lower()
        
        if initial in ['hello', 'hi', 'hey']:
            print("\n👨‍🌾 Hello! Nice to meet you!")
            print("Let's find some schemes for your farm.")
            break
        elif initial in ['yes', 'y', 'start', 'ok']:
            print("\n👨‍🌾 Great! Let's begin.")
            break
        elif initial in ['no', 'n', 'exit', 'quit']:
            print("👨‍🌾 Okay, come back when you're ready!")
            return
        else:
            print("\nLet's begin.")
            break
    
    # Continue with existing code from "while True:" for state question
    while True:
        # State
        while True:
            user_input = input("Which state are you farming in? \n> ").strip()
            result = state_guesser.guess(user_input)
            
            if result and result['confidence'] >= 0.4:
                state = result['word']
                confidence_pct = int(result['confidence'] * 100)
                print(f"✅ Got it! You're farming in {state} (confidence: {confidence_pct}%)")
                break
            else:
                if result:
                    print(f"❌ Not sure (confidence: {int(result['confidence']*100)}%). Please try again.")
                else:
                    print("❌ Couldn't understand. Please try again.")

        # Land + Acres
        while True:
            land_input = input("\nDo you have farmland? (yes/no) \n> ").strip().lower()
            
            if land_input in ['yes', 'y', 'no', 'n']:
                if land_input in ['yes', 'y']:
                    while True:
                        acres_input = input("How many acres? \n> ").strip()
                        acres = extract_number(acres_input)
                        
                        if acres is not None:
                            print(f"✅ {acres} acres noted.")
                            break
                        else:
                            print("❌ Please enter a valid number (e.g., 2.5, 5)")
                else:
                    acres = 0.0
                    print("✅ No farmland noted.")
                break
            else:
                print("❌ Please answer yes or no.")

        # Crops
        while True:
            crops_input = input("\nWhat do you grow or do? (comma separated) \n> ").strip()
            
            # Try to guess the crop from the entire input
            result = crop_guesser.guess(crops_input)
            
            if result and result['confidence'] >= 0.4:
                crops = [result['word']]
                print(f"\n🔍 ML Analysis:")
                print(f"   '{crops_input}' → {result['word']} (confidence: {int(result['confidence']*100)}%)")
                print(f"\n✅ Got it: {crops[0]}")
                break
            else:
                # If whole input fails, try splitting by commas
                items = [c.strip() for c in crops_input.split(',') if c.strip()]
                valid_crops = []
                all_good = True
                
                print("\n🔍 ML Analysis:")
                for item in items:
                    result = crop_guesser.guess(item)
                    if result and result['confidence'] >= 0.4:
                        valid_crops.append(result['word'])
                        print(f"   '{item}' → {result['word']} (confidence: {int(result['confidence']*100)}%)")
                    else:
                        all_good = False
                        if result:
                            print(f"   '{item}' → ❌ Low confidence ({int(result['confidence']*100)}%)")
                        else:
                            print(f"   '{item}' → ❌ Could not recognize")
                
                if all_good and valid_crops:
                    crops = valid_crops
                    print(f"\n✅ Got it: {', '.join(crops)}")
                    break
                else:
                    print("\n❌ Couldn't recognize. Please try again.")

        # Farmer type
        while True:
            type_input = input("\nFarmer type (Individual, SHG, FPO, Other) \n> ").strip()
            result = type_guesser.guess(type_input)
            
            if result and result['confidence'] >= 0.4:
                ftype = result['word']
                confidence_pct = int(result['confidence'] * 100)
                print(f"✅ Farmer type: {ftype} (confidence: {confidence_pct}%)")
                break
            else:
                if result:
                    print(f"❌ Not sure (confidence: {int(result['confidence']*100)}%). Please try again.")
                else:
                    print("❌ Couldn't understand. Please try again.")

        # Create farmer dict and find schemes
        farmer = {"state": state, "acres": acres, "crops": crops, "type": ftype}

        print("\n" + "=" * 50)
        print("🔍 Finding matching schemes...")
        print("=" * 50)

        matches = find_eligible_schemes(schemes, farmer)
        rank_and_display_schemes(matches, farmer)  # Changed from pretty_print_schemes

        # Check another
        again = input("\nCheck another farmer? (yes/no) \n> ").strip().lower()
        if again not in ['yes', 'y']:
            print("\n👋 Thank you! Goodbye!")
            break


# ==================== MAIN ====================

if __name__ == "__main__":
    if not os.path.exists("schemes.json"):
        print("❌ schemes.json not found. Please ensure the file exists.")
    else:
        run()
        
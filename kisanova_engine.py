import os
import json
import re
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

# Import state list and core scheme matcher from chatbot.py
from chatbot import INDIAN_STATES, VALID_CROPS, FARMER_TYPES, find_eligible_schemes, load_schemes

# Try importing groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ==================== PYDANTIC MODELS ====================

class ExtractedFarmerProfile(BaseModel):
    state: Optional[str] = Field(
        default=None, 
        description="Exact Indian State or Union Territory name if explicitly mentioned. Null if missing, ambiguous (e.g. 'South India'), or unresolvable."
    )
    acres: Optional[float] = Field(
        default=None, 
        description="Farmland area in acres as a float. Null if missing/unspecified. Set to 0.0 if farmer explicitly states they have no land / are landless."
    )
    crops: List[str] = Field(
        default_factory=list, 
        description="List of specific crops, crops grown, or agricultural activities (e.g. ['paddy', 'cotton', 'dairy']). Empty if none mentioned."
    )
    farmer_type: Optional[str] = Field(
        default=None, 
        description="Farmer category: 'individual', 'shg' (Self Help Group), 'fpo' (Farmer Producer Org), or 'other'. Null if unspecified."
    )
    income: Optional[float] = Field(
        default=None, 
        description="Annual income in INR if explicitly mentioned, otherwise null."
    )
    is_state_ambiguous: bool = Field(
        default=False, 
        description="Set to True if location is regional or ambiguous (e.g. 'South India', 'my village') and cannot be mapped to a specific Indian state/UT."
    )
    is_empty_or_offtopic: bool = Field(
        default=False, 
        description="Set to True if user input is empty, gibberish, or completely unrelated to agriculture."
    )


# ==================== STATE & CROP NORMALIZATION ====================

STATE_LOOKUP = {s.lower(): s for s in INDIAN_STATES}

AMBIGUOUS_REGIONS = [
    "south india", "north india", "east india", "west india", "northeast", "central india",
    "my state", "my village", "nearby town", "some region", "region", "district"
]

GENERIC_CROP_WORDS = {"crop", "crops", "farm", "farming", "agriculture", "produce", "something", "stuff"}

def normalize_state(raw_state: Optional[str]) -> Tuple[Optional[str], bool]:
    """
    Normalizes raw state input to canonical Indian state name.
    Returns (canonical_state_name, is_ambiguous).
    """
    if not raw_state or not str(raw_state).strip():
        return None, False

    clean_state = str(raw_state).strip().lower()
    
    for region in AMBIGUOUS_REGIONS:
        if region in clean_state:
            return None, True

    if clean_state in STATE_LOOKUP:
        return STATE_LOOKUP[clean_state], False

    for state_lower, state_canonical in STATE_LOOKUP.items():
        if clean_state == state_lower or state_lower in clean_state or clean_state in state_lower:
            return state_canonical, False

    return None, True


def extract_acres_from_text(text: str) -> Optional[float]:
    """Regex helper to extract land acreage from natural language."""
    match = re.search(r'(\d+(?:\.\d+)?)\s*(acres?|acre|hectares?|\bha\b)', text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        unit = match.group(2).lower()
        if 'hectare' in unit or unit == 'ha':
            val = val * 2.47105
        return round(val, 2)
    return None


# ==================== PROVIDER ABSTRACTION ====================

class LLMProvider(ABC):
    """Abstract base class for Kisanova LLM providers."""

    @abstractmethod
    def extract_farmer_profile(self, message: str) -> Dict[str, Any]:
        """Extracts structured farmer profile from natural language message."""
        pass

    @abstractmethod
    def generate_explanation(self, profile: Dict[str, Any], schemes: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Generates personalized explanation string for matched schemes. Returns (text, engine_name)."""
        pass


class LocalFallbackProvider(LLMProvider):
    """Deterministic local rule-based fallback provider."""

    def extract_farmer_profile(self, message: str) -> Dict[str, Any]:
        user_text = (message or "").strip()
        extracted = _fallback_nlp_extraction(user_text)

        norm_state, state_is_ambiguous = normalize_state(extracted.state)
        final_state = None if (state_is_ambiguous or extracted.is_state_ambiguous) else norm_state
        state_ambiguous_flag = state_is_ambiguous or extracted.is_state_ambiguous

        clean_crops = [c for c in (extracted.crops or []) if c.lower().strip() not in GENERIC_CROP_WORDS]

        profile = {
            "state": final_state,
            "acres": extracted.acres,
            "crops": clean_crops,
            "farmer_type": extracted.farmer_type,
            "income": extracted.income
        }

        missing_fields = []
        if state_ambiguous_flag or profile["state"] is None:
            missing_fields.append("state")

        if profile["acres"] is None:
            crop_lowers = [c.lower() for c in profile["crops"]]
            if any(x in crop_lowers for x in ["dairy", "livestock", "poultry", "cattle"]):
                profile["acres"] = 0.0
            else:
                missing_fields.append("acres")

        if missing_fields:
            if state_ambiguous_flag:
                clarification_msg = "Which state or Union Territory are you farming in? Please specify your exact state in India."
            elif "state" in missing_fields and "acres" in missing_fields:
                clarification_msg = "Please tell me your state and approximate landholding in acres so I can find suitable schemes for you."
            elif "state" in missing_fields:
                clarification_msg = "Which state or Union Territory are you farming in?"
            elif "acres" in missing_fields:
                clarification_msg = "How many acres of farmland do you have?"
            else:
                clarification_msg = f"Please provide your {', '.join(missing_fields)}."

            return {
                "status": "needs_clarification",
                "engine_used": "fallback",
                "farmer_profile": profile,
                "missing_fields": missing_fields,
                "message": clarification_msg
            }

        return {
            "status": "success",
            "engine_used": "fallback",
            "farmer_profile": profile
        }

    def generate_explanation(self, profile: Dict[str, Any], schemes: List[Dict[str, Any]]) -> Tuple[str, str]:
        if not schemes:
            state_str = profile.get('state') or 'your state'
            acres_str = f"{profile.get('acres')} acres" if profile.get('acres') is not None else ""
            crops_str = ", ".join(profile.get('crops', []))
            return (
                f"Based on your profile in **{state_str}** ({acres_str}, crops: {crops_str}), "
                "we could not find any specific matching government schemes in our dataset at this time. "
                "Please check back as new schemes are added regularly.",
                "fallback"
            )
        return _fallback_explanation_generator(profile, schemes), "fallback"


class GroqProvider(LLMProvider):
    """Groq LLM provider for structured extraction and natural language generation."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.client = None
        if GROQ_AVAILABLE and self.api_key and len(self.api_key.strip()) > 5:
            try:
                self.client = Groq(api_key=self.api_key.strip())
            except Exception as e:
                logger.warning("Failed to initialize Groq client: %s", e)
                self.client = None
        self.fallback = LocalFallbackProvider()

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Safely extract a JSON object from model output.
        Handles Qwen3-style <think>...</think> reasoning blocks,
        markdown code fences, and leading/trailing whitespace.
        """
        # Strip thinking blocks (Qwen3 and similar models)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        # Strip markdown code fences
        content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE).strip()
        # Extract JSON object
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(content)

    def extract_farmer_profile(self, message: str) -> Dict[str, Any]:
        if not message or not message.strip():
            return self.fallback.extract_farmer_profile(message)

        if not self.client:
            logger.info("Groq API key not configured or client invalid. Using Fallback Engine.")
            return self.fallback.extract_farmer_profile(message)

        user_text = message.strip()
        try:
            system_prompt = """
            You are an AI profile extraction assistant for Kisanova, an Indian farmer scheme discovery platform.
            Analyze the farmer's natural-language query and extract structured JSON fields.

            JSON SCHEMA OUTPUT REQUIREMENT:
            Return ONLY a valid JSON object matching this structure:
            {
              "state": string or null (Exact Indian State/UT name. Set to null if vague e.g. 'South India' or unmentioned),
              "acres": number or null (Landholding area in acres. Set to 0.0 for landless dairy/livestock farmers),
              "crops": list of strings (Crops or agricultural activities mentioned. Empty list [] if none),
              "farmer_type": string or null ('individual', 'shg', 'fpo', or 'other'),
              "income": number or null,
              "is_state_ambiguous": boolean (true if location mentioned is regional e.g. 'South India' or 'my village'),
              "is_empty_or_offtopic": boolean
            }

            CRITICAL RULES:
            - NEVER invent missing information. If a field is not specified, return null.
            - Do NOT guess ambiguous locations.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Farmer Query: \"{user_text}\""}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            response_content = response.choices[0].message.content or ""
            extracted_json = self._parse_json_response(response_content)
            extracted = ExtractedFarmerProfile(**extracted_json)

            # Normalize state & crops
            norm_state, state_is_ambiguous = normalize_state(extracted.state)
            final_state = None if (state_is_ambiguous or extracted.is_state_ambiguous) else norm_state
            state_ambiguous_flag = state_is_ambiguous or extracted.is_state_ambiguous

            clean_crops = [c for c in (extracted.crops or []) if c.lower().strip() not in GENERIC_CROP_WORDS]

            profile = {
                "state": final_state,
                "acres": extracted.acres,
                "crops": clean_crops,
                "farmer_type": extracted.farmer_type,
                "income": extracted.income
            }

            missing_fields = []
            if state_ambiguous_flag or profile["state"] is None:
                missing_fields.append("state")

            if profile["acres"] is None:
                crop_lowers = [c.lower() for c in profile["crops"]]
                if any(x in crop_lowers for x in ["dairy", "livestock", "poultry", "cattle"]):
                    profile["acres"] = 0.0
                else:
                    missing_fields.append("acres")

            if missing_fields:
                if state_ambiguous_flag:
                    clarification_msg = "Which state or Union Territory are you farming in? Please specify your exact state in India."
                elif "state" in missing_fields and "acres" in missing_fields:
                    clarification_msg = "Please tell me your state and approximate landholding in acres so I can find suitable schemes for you."
                elif "state" in missing_fields:
                    clarification_msg = "Which state or Union Territory are you farming in?"
                elif "acres" in missing_fields:
                    clarification_msg = "How many acres of farmland do you have?"
                else:
                    clarification_msg = f"Please provide your {', '.join(missing_fields)}."

                return {
                    "status": "needs_clarification",
                    "engine_used": "groq",
                    "farmer_profile": profile,
                    "missing_fields": missing_fields,
                    "message": clarification_msg
                }

            return {
                "status": "success",
                "engine_used": "groq",
                "farmer_profile": profile
            }

        except Exception as e:
            logger.warning("Groq API extraction failed (%s). Falling back to LocalFallbackProvider.", e)
            return self.fallback.extract_farmer_profile(message)

    def generate_explanation(self, profile: Dict[str, Any], schemes: List[Dict[str, Any]]) -> Tuple[str, str]:
        if not schemes:
            return self.fallback.generate_explanation(profile, schemes)

        if not self.client:
            return self.fallback.generate_explanation(profile, schemes)

        try:
            # Trim to top 5 schemes and only essential fields to stay within TPM limits
            trimmed_schemes = []
            for s in schemes[:5]:
                trimmed_schemes.append({
                    "name": s.get("name"),
                    "benefit": s.get("benefit"),
                    "eligibility": s.get("eligibility"),
                    "docs": s.get("docs", [])[:4],
                    "url": s.get("url"),
                    "category": s.get("category"),
                })

            crops_str = ", ".join(profile.get("crops", [])) or "not specified"
            schemes_json = json.dumps(trimmed_schemes, indent=2)

            system_prompt = (
                "You are Kisanova's AI Assistant for Indian Farmers.\n"
                "Provide a warm, clear, structured, and personalized explanation of the matched government schemes.\n\n"
                f"FARMER PROFILE:\n"
                f"- State: {profile.get('state')}\n"
                f"- Landholding: {profile.get('acres')} acres\n"
                f"- Crops / Activities: {crops_str}\n"
                f"- Farmer Type: {profile.get('farmer_type') or 'Individual'}\n\n"
                f"MATCHED GOVERNMENT SCHEMES:\n{schemes_json}\n\n"
                "RESPONSE INSTRUCTIONS:\n"
                "1. Warmly greet the farmer and summarize why these schemes fit their profile.\n"
                "2. For each scheme: name, why it fits, key benefit, required docs, and apply URL.\n"
                "3. STRICT GROUNDING: Use ONLY the data provided. Never invent amounts or terms.\n"
                "4. End with: '📌 *Note: Scheme matches indicate potential relevance. Final eligibility is subject to official government verification.*'"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the personalized scheme explanation for this farmer."}
                ],
                temperature=0.3,
                max_tokens=1024
            )

            explanation_text = (response.choices[0].message.content or "").strip()
            return explanation_text, "groq"

        except Exception as e:
            logger.warning("Groq API explanation generation failed (%s). Falling back to LocalFallbackProvider.", e)
            return self.fallback.generate_explanation(profile, schemes)


# Helper function to get active provider
def get_provider() -> LLMProvider:
    api_key = os.getenv("GROQ_API_KEY")
    if GROQ_AVAILABLE and api_key and len(api_key.strip()) > 5:
        return GroqProvider(api_key=api_key)
    return LocalFallbackProvider()


# ==================== STEP 1: PROFILE EXTRACTION ROUTER ====================

def extract_farmer_profile(message: str) -> Dict[str, Any]:
    """
    Extracts structured profile fields using active LLMProvider (GroqProvider or LocalFallbackProvider).
    """
    provider = get_provider()
    return provider.extract_farmer_profile(message)


# ==================== STEP 2: DETERMINISTIC MATCHING (100% PYTHON) ====================

def find_kisanova_schemes(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes 100% deterministic scheme matching using python logic.
    GROQ IS NEVER USED TO DECIDE ELIGIBILITY.
    """
    schemes = load_schemes()
    matches = find_eligible_schemes(schemes, profile)
    return matches


# ==================== STEP 3: EXPLANATION GENERATOR ROUTER ====================

def generate_scheme_explanation(profile: Dict[str, Any], schemes: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Generates personalized scheme explanation using active LLMProvider.
    Returns (explanation_text, engine_used).
    """
    provider = get_provider()
    return provider.generate_explanation(profile, schemes)


# ==================== FALLBACK HELPERS ====================

def _fallback_nlp_extraction(text: str) -> ExtractedFarmerProfile:
    """Fallback rule-based NLP extraction when Groq API is offline/unavailable."""
    text_lower = text.lower()

    detected_state = None
    is_ambiguous = False
    for region in AMBIGUOUS_REGIONS:
        if region in text_lower:
            is_ambiguous = True
            break

    if not is_ambiguous:
        for s in INDIAN_STATES:
            if s.lower() in text_lower:
                detected_state = s
                break

    acres = extract_acres_from_text(text)

    crops = []
    for crop in VALID_CROPS:
        if crop.lower() in text_lower:
            crops.append(crop)
    if ("paddy" in text_lower or "rice" in text_lower) and "rice" not in crops:
        crops.append("rice")

    farmer_type = None
    if "shg" in text_lower or "self help" in text_lower:
        farmer_type = "shg"
    elif "fpo" in text_lower or "producer org" in text_lower:
        farmer_type = "fpo"
    elif "individual" in text_lower or "small farmer" in text_lower:
        farmer_type = "individual"

    return ExtractedFarmerProfile(
        state=detected_state,
        acres=acres,
        crops=crops,
        farmer_type=farmer_type,
        income=None,
        is_state_ambiguous=is_ambiguous,
        is_empty_or_offtopic=False
    )


def _fallback_explanation_generator(profile: Dict[str, Any], schemes: List[Dict[str, Any]]) -> str:
    """Fallback generator ensuring a complete response even without Groq API."""
    state = profile.get("state", "your state")
    acres = profile.get("acres", 0)
    crops_str = ", ".join(profile.get("crops", [])) or "your crops"

    lines = [
        f"🌾 **Great news! We found {len(schemes)} government scheme(s) matching your profile in {state} ({acres} acres, {crops_str}):**\n"
    ]

    for i, s in enumerate(schemes[:4], 1):
        name = s.get("name", "Government Scheme")
        benefit = s.get("benefit", "N/A")
        docs = ", ".join(s.get("docs", [])) if s.get("docs") else "Standard ID & Land documents"
        url = s.get("url", "#")
        eligibility = s.get("eligibility", "")

        lines.append(f"### {i}. {name}")
        lines.append(f"• **Why it's relevant**: Matches your profile in {state} with cultivable land / crop eligibility.")
        lines.append(f"• **Key Benefit**: {benefit}")
        lines.append(f"• **Eligibility**: {eligibility}")
        lines.append(f"• **Required Documents**: {docs}")
        lines.append(f"• **Apply Here**: [{url}]({url})\n")

    lines.append("📌 *Note: Scheme matches indicate potential relevance based on your profile. Final eligibility and approval are subject to official government verification.*")
    return "\n".join(lines)

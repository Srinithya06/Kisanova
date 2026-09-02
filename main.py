from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import logging
import time
from datetime import datetime
import uuid
from functools import wraps


# Import your chatbot logic
from chatbot import (
    state_guesser, crop_guesser, type_guesser,
    find_eligible_schemes, rank_and_display_schemes,
    load_schemes, extract_number, INDIAN_STATES,
    VALID_CROPS, FARMER_TYPES
)

from kisanova_engine import (
    extract_farmer_profile,
    find_kisanova_schemes,
    generate_scheme_explanation
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Configure CORS
CORS(app, origins=["*"])  # In production, replace with specific origins

# ==================== Pydantic Models ====================

class ChatMessage(BaseModel):
    """Model for chat messages"""
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class FarmerDetails(BaseModel):
    """Model for farmer details"""
    state: str
    acres: float
    crops: List[str]
    farmer_type: str
    session_id: Optional[str] = None

class SchemeResponse(BaseModel):
    """Model for scheme response"""
    id: str
    name: str
    benefit: str
    docs: List[str]
    url: Optional[str] = None
    relevance_score: int
    relevance_label: str

class ChatResponse(BaseModel):
    """Model for chat response"""
    session_id: str
    message: str
    type: str  # question, answer, scheme_result, etc.
    options: Optional[List[str]] = None
    schemes: Optional[List[SchemeResponse]] = None
    confidence: Optional[float] = None
    suggested_input: Optional[str] = None

class GuessResponse(BaseModel):
    """Model for ML guess response"""
    original_input: str
    guessed_word: str
    confidence: float
    alternatives: Optional[List[Dict[str, Any]]] = None

class EligibilityRequest(BaseModel):
    """Model for eligibility check request"""
    farmer: FarmerDetails

# ==================== Session Management ====================

class SessionManager:
    """Manage user sessions"""
    def __init__(self):
        self.sessions = {}
        self.session_timeout = 3600  # 1 hour timeout
    
    def create_session(self) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "context": {
                "step": "start",
                "farmer": {},
                "history": []
            }
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        if session_id in self.sessions:
            session_data = self.sessions[session_id]
            # Check timeout
            if (datetime.now() - session_data["last_active"]).seconds > self.session_timeout:
                del self.sessions[session_id]
                return None
            session_data["last_active"] = datetime.now()
            return session_data
        return None
    
    def update_session(self, session_id: str, context: Dict):
        """Update session context"""
        if session_id in self.sessions:
            self.sessions[session_id]["context"] = context
            self.sessions[session_id]["last_active"] = datetime.now()
            return True
        return False
    
    def add_to_history(self, session_id: str, message: str, response: str):
        """Add interaction to history"""
        if session_id in self.sessions:
            if "history" not in self.sessions[session_id]["context"]:
                self.sessions[session_id]["context"]["history"] = []
            self.sessions[session_id]["context"]["history"].append({
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "response": response
            })

session_manager = SessionManager()

# Load schemes
try:
    schemes = load_schemes()
    logger.info(f"✅ Loaded {len(schemes)} schemes successfully")
except Exception as e:
    logger.error(f"❌ Failed to load schemes: {e}")
    schemes = {}

# ==================== Helper Functions ====================

def validate_json(schema_class):
    """Decorator to validate JSON request body against Pydantic model"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            try:
                data = schema_class(**request.get_json())
                return f(data, *args, **kwargs)
            except ValidationError as e:
                return jsonify({"error": e.errors()}), 400
        return decorated_function
    return decorator

def get_json_data():
    """Get JSON data from request"""
    if not request.is_json:
        return None
    return request.get_json()

# ==================== API Endpoints ====================

@app.route('/', methods=['GET'])
def root():
    """Serve the Kisanova web UI"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "schemes_loaded": len(schemes),
        "ml_models_loaded": all([
            state_guesser.is_trained,
            crop_guesser.is_trained,
            type_guesser.is_trained
        ])
    })

@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Create a new chat session"""
    session_id = session_manager.create_session()
    return jsonify({
        "session_id": session_id,
        "message": "Session created successfully",
        "next_step": "state",
        "question": "Which state are you farming in?"
    })

@app.route('/api/chat', methods=['POST'])
@validate_json(ChatMessage)
def chat(validated_data):
    """Main chat endpoint for interactive conversation"""
    
    # Create or get session
    session_id = validated_data.session_id
    if not session_id:
        session_id = session_manager.create_session()
        session_data = session_manager.get_session(session_id)
        context = session_data["context"]
    else:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            # Session expired or invalid
            session_id = session_manager.create_session()
            session_data = session_manager.get_session(session_id)
            context = session_data["context"]
        else:
            context = session_data["context"]
    
    user_input = validated_data.message.strip()
    
    # Add to history
    session_manager.add_to_history(session_id, user_input, "")
    
    # Process based on current step
    step = context.get("step", "start")
    farmer = context.get("farmer", {})
    
    response = {
        "session_id": session_id,
        "message": "",
        "type": "question"
    }
    
    if step == "start":
        # Initial greeting
        response["message"] = "👋 Hi! I'm your farming scheme assistant. Which state are you farming in?"
        response["type"] = "question"
        context["step"] = "state"
        
    elif step == "state":
        # Process state input
        result = state_guesser.guess(user_input)
        if result and result['confidence'] >= 0.4:
            farmer["state"] = result['word']
            response["message"] = f"✅ Got it! You're farming in {result['word']}. Do you have farmland? (yes/no)"
            response["type"] = "question"
            response["options"] = ["yes", "no"]
            response["confidence"] = result['confidence']
            context["step"] = "land"
        else:
            confidence = result['confidence'] if result else 0
            response["message"] = f"❌ Not sure (confidence: {int(confidence*100)}%). Please try again. Which state are you farming in?"
            response["type"] = "question"
            if result:
                response["suggested_input"] = f"Did you mean {result['word']}?"
    
    elif step == "land":
        # Process land question
        if user_input.lower() in ['yes', 'y']:
            response["message"] = "How many acres do you have?"
            response["type"] = "question"
            context["step"] = "acres"
        elif user_input.lower() in ['no', 'n']:
            farmer["acres"] = 0.0
            response["message"] = "✅ No farmland noted. What do you grow or do? (e.g., wheat, rice, dairy)"
            response["type"] = "question"
            context["step"] = "crops"
        else:
            response["message"] = "❌ Please answer yes or no. Do you have farmland?"
            response["type"] = "question"
            response["options"] = ["yes", "no"]
    
    elif step == "acres":
        # Process acres input
        acres = extract_number(user_input)
        if acres is not None:
            farmer["acres"] = acres
            response["message"] = f"✅ {acres} acres noted. What do you grow or do? (e.g., wheat, rice, dairy)"
            response["type"] = "question"
            context["step"] = "crops"
        else:
            response["message"] = "❌ Please enter a valid number (e.g., 2.5, 5). How many acres?"
            response["type"] = "question"
    
    elif step == "crops":
        # Process crops input
        crops = []
        items = [c.strip() for c in user_input.split(',') if c.strip()]
        
        for item in items:
            result = crop_guesser.guess(item)
            if result and result['confidence'] >= 0.4:
                crops.append(result['word'])
        
        if crops:
            farmer["crops"] = crops
            response["message"] = f"✅ Got it: {', '.join(crops)}. What is your farmer type? (Individual, SHG, FPO, Other)"
            response["type"] = "question"
            response["options"] = FARMER_TYPES
            context["step"] = "type"
        else:
            response["message"] = "❌ Couldn't recognize crops. Please try again (comma separated):"
            response["type"] = "question"
    
    elif step == "type":
        # Process farmer type
        result = type_guesser.guess(user_input)
        if result and result['confidence'] >= 0.4:
            farmer["type"] = result['word']
            
            # All details collected, find schemes
            matches = find_eligible_schemes(schemes, farmer)
            
            # Format schemes for response
            scheme_list = []
            for scheme in matches:
                scheme_list.append({
                    "id": scheme.get('id', ''),
                    "name": scheme.get('name', ''),
                    "benefit": scheme.get('benefit', ''),
                    "docs": scheme.get('docs', []),
                    "url": scheme.get('url'),
                    "relevance_score": 0,  # You can add scoring logic here
                    "relevance_label": "Eligible"
                })
            
            response["message"] = f"✅ Farmer type: {result['word']}. Found {len(matches)} matching schemes!"
            response["type"] = "scheme_result"
            response["schemes"] = scheme_list
            context["step"] = "complete"
        else:
            confidence = result['confidence'] if result else 0
            response["message"] = f"❌ Not sure (confidence: {int(confidence*100)}%). Please select farmer type:"
            response["type"] = "question"
            response["options"] = FARMER_TYPES
    
    elif step == "complete":
        # Conversation complete, offer to start over
        if user_input.lower() in ['yes', 'y', 'start over', 'new']:
            response["message"] = "Great! Let's start over. Which state are you farming in?"
            response["type"] = "question"
            context["step"] = "state"
            context["farmer"] = {}
        else:
            response["message"] = "Thank you for using Farmer Scheme Assistant! Type 'start over' to check another farmer."
            response["type"] = "end"
    
    # Update context
    context["farmer"] = farmer
    session_manager.update_session(session_id, context)
    
    return jsonify(response)

@app.route('/api/guess/<field>', methods=['POST'])
def guess_field(field):
    """ML-powered guess for a specific field"""
    data = get_json_data()
    if not data:
        return jsonify({"error": "Request must be JSON"}), 400
    
    user_input = data.get("input", "")
    
    if not user_input:
        return jsonify({"error": "Input is required"}), 400
    
    guesser_map = {
        "state": state_guesser,
        "crop": crop_guesser,
        "type": type_guesser
    }
    
    valid_values_map = {
        "state": INDIAN_STATES,
        "crop": VALID_CROPS,
        "type": FARMER_TYPES
    }
    
    if field not in guesser_map:
        return jsonify({"error": f"Invalid field. Choose from: {list(guesser_map.keys())}"}), 400
    
    guesser = guesser_map[field]
    result = guesser.guess(user_input)
    
    if not result:
        return jsonify({"error": "Could not guess the input"}), 404
    
    # Get alternatives (simple implementation)
    alternatives = []
    for word in valid_values_map[field][:3]:  # Get first 3 valid values
        if word.lower() != result['word'].lower():
            alt_result = guesser.guess(word)
            if alt_result and alt_result['confidence'] > 0.3:
                alternatives.append({
                    "word": word,
                    "confidence": alt_result['confidence']
                })
    
    response = {
        "original_input": user_input,
        "guessed_word": result['word'],
        "confidence": result['confidence'],
        "alternatives": alternatives[:3]
    }
    
    return jsonify(response)

@app.route('/api/check-eligibility', methods=['POST'])
@validate_json(EligibilityRequest)
def check_eligibility(validated_data):
    """Check eligibility for schemes based on farmer details"""
    farmer = validated_data.farmer.dict()
    
    # Convert crops list to match expected format
    farmer["crops"] = [crop.lower() for crop in farmer["crops"]]
    
    # Find eligible schemes
    matches = find_eligible_schemes(schemes, farmer)
    
    # Format response
    eligible_schemes = []
    for scheme in matches:
        eligible_schemes.append({
            "id": scheme.get('id'),
            "name": scheme.get('name'),
            "benefit": scheme.get('benefit'),
            "docs": scheme.get('docs', []),
            "url": scheme.get('url'),
            "params": scheme.get('params', {})
        })
    
    return jsonify({
        "farmer": farmer,
        "total_schemes": len(eligible_schemes),
        "schemes": eligible_schemes
    })

@app.route('/api/schemes', methods=['GET'])
def get_all_schemes():
    """Get all available schemes"""
    scheme_list = []
    for scheme_id, scheme in schemes.items():
        scheme_list.append({
            "id": scheme_id,
            "name": scheme.get('name'),
            "benefit": scheme.get('benefit'),
            "params": scheme.get('params', {})
        })
    
    return jsonify({
        "total": len(scheme_list),
        "schemes": scheme_list
    })

@app.route('/api/schemes/<scheme_id>', methods=['GET'])
def get_scheme(scheme_id):
    """Get specific scheme by ID"""
    scheme = schemes.get(scheme_id)
    if not scheme:
        return jsonify({"error": "Scheme not found"}), 404
    return jsonify(scheme)

@app.route('/api/valid-values/<field>', methods=['GET'])
def get_valid_values(field):
    """Get valid values for a specific field"""
    valid_values = {
        "states": INDIAN_STATES,
        "crops": VALID_CROPS,
        "types": FARMER_TYPES
    }
    
    if field not in valid_values:
        return jsonify({"error": f"Invalid field. Choose from: {list(valid_values.keys())}"}), 400
    
    return jsonify({
        "field": field,
        "values": valid_values[field]
    })

# ==================== Kisanova API Endpoint ====================

@app.route('/api/kisanova/assist', methods=['POST'])
def kisanova_assist():
    """
    Stateless Kisanova natural-language scheme assistant endpoint.
    Processes full farmer query in a single shot without SessionManager.
    """
    data = get_json_data()
    if not data or "message" not in data:
        return jsonify({
            "success": False,
            "error": "Request must be JSON with a 'message' string field"
        }), 400

    user_message = str(data.get("message", "")).strip()

    # Step 1: Extract farmer profile & check missing/ambiguous required fields
    extraction_result = extract_farmer_profile(user_message)
    extraction_engine = extraction_result.get("engine_used", "fallback")

    if extraction_result.get("status") == "needs_clarification":
        return jsonify({
            "success": True,
            "status": "needs_clarification",
            "engine": {
                "extraction_engine": extraction_engine,
                "explanation_engine": "none",
                "live_groq_active": (extraction_engine == "groq")
            },
            "farmer_profile": extraction_result.get("farmer_profile"),
            "missing_fields": extraction_result.get("missing_fields", []),
            "message": extraction_result.get("message", "Please provide more information.")
        })

    profile = extraction_result.get("farmer_profile", {})

    # Step 2: Deterministic Eligibility Matching (Python ONLY - Groq NEVER decides eligibility)
    matched_schemes = find_kisanova_schemes(profile)

    formatted_schemes = []
    for scheme in matched_schemes:
        formatted_schemes.append({
            "id": scheme.get("id"),
            "name": scheme.get("name"),
            "benefit": scheme.get("benefit"),
            "docs": scheme.get("docs", []),
            "url": scheme.get("url"),
            "category": scheme.get("category"),
            "eligibility": scheme.get("eligibility"),
            "params": scheme.get("params", {})
        })

    # Step 3: Groq Explanation Generation
    explanation_text, explanation_engine = generate_scheme_explanation(profile, matched_schemes)

    # Log pipeline execution status
    logger.info("🤖 Kisanova Assist | Extraction Engine: %s | Explanation Engine: %s", extraction_engine, explanation_engine)

    return jsonify({
        "success": True,
        "status": "success",
        "engine": {
            "extraction_engine": extraction_engine,
            "explanation_engine": explanation_engine,
            "live_groq_active": (extraction_engine == "groq" and explanation_engine == "groq")
        },
        "farmer_profile": profile,
        "total_schemes": len(formatted_schemes),
        "schemes": formatted_schemes,
        "answer": explanation_text
    })

# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "status_code": 404}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed", "status_code": 405}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error", "status_code": 500}), 500

# ==================== Run the API ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("ENV") == "development"

    print(f"[Kisanova] Starting Farmer Scheme Assistant API with Flask...")
    print(f"[Kisanova] UI: http://localhost:{port}/")
    print(f"[Kisanova] Health: http://localhost:{port}/health")

    app.run(host=host, port=port, debug=debug)
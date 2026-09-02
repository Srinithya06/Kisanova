import unittest
import json
import os
from unittest.mock import MagicMock, patch
from main import app
from chatbot import INDIAN_STATES, find_eligible_schemes, load_schemes
from kisanova_engine import (
    extract_farmer_profile,
    find_kisanova_schemes,
    generate_scheme_explanation,
    GroqProvider,
    LocalFallbackProvider,
    LLMProvider
)

class TestKisanovaGroqImplementation(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_provider_abstraction_hierarchy(self):
        self.assertTrue(issubclass(GroqProvider, LLMProvider))
        self.assertTrue(issubclass(LocalFallbackProvider, LLMProvider))

    def test_deduplicated_find_eligible_schemes(self):
        schemes = load_schemes()
        farmer = {"state": "Maharashtra", "acres": 3.0, "crops": ["wheat"], "type": "individual"}
        matches = find_eligible_schemes(schemes, farmer)
        self.assertIsInstance(matches, list)
        self.assertGreater(len(matches), 0)

    def test_expanded_state_coverage(self):
        self.assertIn("Sikkim", INDIAN_STATES)
        self.assertIn("Telangana", INDIAN_STATES)
        self.assertIn("Ladakh", INDIAN_STATES)
        self.assertEqual(len(INDIAN_STATES), 36)

    def test_groq_provider_fallback_when_key_missing(self):
        with unittest.mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            provider = GroqProvider(api_key="")
            result = provider.extract_farmer_profile("I am a farmer in Maharashtra with 5 acres growing wheat.")
        self.assertEqual(result.get("status"), "success")
        self.assertEqual(result.get("engine_used"), "fallback")

    def test_mock_groq_extraction_and_explanation(self):
        mock_groq_response = MagicMock()
        mock_groq_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "state": "Telangana",
                "acres": 3.0,
                "crops": ["paddy"],
                "farmer_type": None,
                "income": None,
                "is_state_ambiguous": False,
                "is_empty_or_offtopic": False
            })))
        ]

        mock_explanation_response = MagicMock()
        mock_explanation_response.choices = [
            MagicMock(message=MagicMock(content="Mocked Groq Explanation text."))
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [mock_groq_response, mock_explanation_response]

        provider = GroqProvider(api_key="gsk_mock_test_key", model="llama-3.3-70b-versatile")
        provider.client = mock_client

        # Test Mocked Extraction
        extract_res = provider.extract_farmer_profile("I am a farmer from Telangana with 3 acres growing paddy.")
        self.assertEqual(extract_res.get("status"), "success")
        self.assertEqual(extract_res.get("engine_used"), "groq")
        self.assertEqual(extract_res["farmer_profile"]["state"], "Telangana")

        # Test Mocked Explanation
        schemes = load_schemes()
        matched = find_eligible_schemes(schemes, extract_res["farmer_profile"])
        expl_text, expl_engine = provider.generate_explanation(extract_res["farmer_profile"], matched)
        self.assertEqual(expl_engine, "groq")
        self.assertEqual(expl_text, "Mocked Groq Explanation text.")

    def test_case_1_telangana_paddy(self):
        payload = {"message": "I'm a farmer from Telangana with 3 acres and I grow paddy."}
        response = self.app.post('/api/kisanova/assist', data=json.dumps(payload), content_type='application/json')
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("status"), "success")
        self.assertIn("engine", data)
        self.assertEqual(data["farmer_profile"]["state"], "Telangana")
        self.assertEqual(data["farmer_profile"]["acres"], 3.0)
        self.assertGreater(len(data["schemes"]), 0)

    def test_case_2_maharashtra_wheat_cotton(self):
        payload = {"message": "I have 5 acres in Maharashtra and grow wheat and cotton."}
        response = self.app.post('/api/kisanova/assist', data=json.dumps(payload), content_type='application/json')
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data["farmer_profile"]["state"], "Maharashtra")
        self.assertEqual(data["farmer_profile"]["acres"], 5.0)

    def test_case_3_karnataka_missing_acres(self):
        payload = {"message": "I'm a farmer from Karnataka."}
        response = self.app.post('/api/kisanova/assist', data=json.dumps(payload), content_type='application/json')
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("status"), "needs_clarification")
        self.assertIn("acres", data.get("missing_fields", []))

    def test_case_4_south_india_ambiguous(self):
        payload = {"message": "I grow some crops in South India."}
        response = self.app.post('/api/kisanova/assist', data=json.dumps(payload), content_type='application/json')
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("status"), "needs_clarification")
        self.assertIn("state", data.get("missing_fields", []))

    def test_case_5_empty_query(self):
        payload = {"message": ""}
        response = self.app.post('/api/kisanova/assist', data=json.dumps(payload), content_type='application/json')
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("status"), "needs_clarification")

    def test_backward_compatibility_endpoints(self):
        res_health = self.app.get('/health')
        self.assertEqual(res_health.status_code, 200)
        res_schemes = self.app.get('/api/schemes')
        self.assertEqual(res_schemes.status_code, 200)

if __name__ == '__main__':
    unittest.main()

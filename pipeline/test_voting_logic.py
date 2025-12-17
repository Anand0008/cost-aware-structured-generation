
import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

# Add pipeline root to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from init_09_voting_engine import VotingEngine, IncompleteLLMResponseError

class TestHybridConsensus(unittest.TestCase):
    
    def setUp(self):
        self.configs = {
            'models_config': {
                'models': {
                    'gemini_2.5_pro': {'model_id': 'gemini-2.5-flash'}
                }
            }
        }
        self.clients = {
            'google_genai': MagicMock()
        }
        self.engine = VotingEngine(self.configs, self.clients)
        
        # Sample Weights
        self.weights = {"ModelA": 0.5, "ModelB": 0.3, "ModelC": 0.2}

    def test_red_line_trigger(self):
        """Test that is_correct=False acts as a Red Line"""
        responses = {
            "ModelA": {"response_json": {"tier_1_core_research": {"answer_validation": {"is_correct": True}}}},
            "ModelB": {"response_json": {"tier_1_core_research": {"answer_validation": {"is_correct": False}}}}, # The sabotuer
        }
        result = self.engine.vote_on_responses(responses, self.weights, {"question_id": "Q1"})
        
        self.assertEqual(result['disputed_fields'], ["RED_LINE_FAILURE"])
        self.assertEqual(result['consensus_score'], 0.0)

    def test_weighted_average(self):
        """Test numeric weighted averaging"""
        responses = {
            "ModelA": {"response_json": {"tier_0_classification": {"difficulty_score": 5}}}, # 0.5
            "ModelB": {"response_json": {"tier_0_classification": {"difficulty_score": 10}}}, # 0.3
            "ModelC": {"response_json": {"tier_0_classification": {"difficulty_score": 0}}}  # 0.2
        }
        # Weighted sum: (5*0.5) + (10*0.3) + (0*0.2) = 2.5 + 3.0 + 0 = 5.5
        # Total weight: 1.0
        # Avg: 5.5
        
        # We need to mock the Red Line check to pass (assume True if missing, or mock it)
        # The logic expects tier_1_...is_correct. If missing, it might return None.
        # Let's patch _check_red_line to skip it for this test
        with patch.object(self.engine, '_check_red_line', return_value=False):
             # Also mock synthesis to avoid calling LLM
             with patch.object(self.engine, '_call_synthesizer_with_retry', return_value={}):
                result = self.engine.vote_on_responses(responses, self.weights, {"question_id": "Q1"})
                
                # Check averaged stats
                expected_val = 5.5
                actual_val = result['converged_fields'].get('tier_0_classification.difficulty_score')
                self.assertEqual(actual_val, expected_val)

    def test_synthesis_audit_success(self):
        """Test that synthesis works when keys match"""
        batch_input = {"tier_1": "data"}
        llm_response = {"tier_1": "synthesized data"}
        
        # Should not raise
        result = self.engine._audit_response(batch_input.keys(), llm_response)
        self.assertTrue(result)

    def test_synthesis_audit_failure(self):
        """Test that synthesis raises error when keys missing"""
        batch_input = {"tier_1": "data", "tier_2": "data"}
        llm_response = {"tier_1": "synthesized data"} # Missing tier_2
        
        with self.assertRaises(IncompleteLLMResponseError):
            self.engine._audit_response(batch_input.keys(), llm_response)

if __name__ == '__main__':
    unittest.main()

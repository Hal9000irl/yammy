# specialist_sales_services.py
# Contains services for generic sales skills, real estate knowledge, and sales agent orchestration.

import time # For simulation
import os # For os.getenv

from config_utils import resolve_config_value


class GenericSalesSkillService:
    """
    Provides foundational sales skills (openings, objection handling, closing).
    Trained on general sales data.
    """
    def __init__(self, service_config: dict):
        self.config = service_config

        raw_model_path = self.config.get('model_path', 'default_sales_model.pkl')
        self.model_path = resolve_config_value(raw_model_path, default_if_placeholder_not_set='default_sales_model.pkl')

        playbooks_config = self.config.get('playbooks', {})
        self.playbooks = {
            "opening_scripts_path": resolve_config_value(
                playbooks_config.get('opening_scripts_path'),
                "path/to/data/sales_training/opening_lines.csv"
            ),
            "objection_handling_path": resolve_config_value(
                playbooks_config.get('objection_handling_path'),
                "path/to/data/sales_training/objection_responses.json"
            ),
            "closing_techniques_path": resolve_config_value(
                playbooks_config.get('closing_techniques_path'),
                "path/to/data/sales_training/closing_playbook.txt"
            )
        }
        print(f"GenericSalesSkillService Initialized (Model: {self.model_path}, Playbook Paths: {self.playbooks})")

    def get_opening_line(self, prospect_profile: dict = None) -> str:
        print("GenericSalesSkills: Getting opening line.")
        if prospect_profile and prospect_profile.get('type') == 'cold_lead':
            return "Hi, this is Alex calling from Premier Properties. I came across your information and wanted to see if you might be considering a move in the near future?"
        return "Thanks for taking my call today. I'm calling from Premier Properties, and I wanted to briefly discuss how we might be able to assist with your real estate needs."

    def handle_objection(self, objection_text: str, sales_context: dict) -> str:
        stage = sales_context.get('stage', 'unknown')
        print(f"GenericSalesSkills: Handling objection '{objection_text}' in sales stage '{stage}'")
        objection_lower = objection_text.lower()
        if "expensive" in objection_lower or "cost too much" in objection_lower:
            return "I understand that budget is a very important consideration. To make sure we're on the same page, could you share what price range you were anticipating, or what aspects of the value are most critical for you?"
        return "That's a valid point. Let me share some information that might provide a different perspective on that..."

    def suggest_closing_technique(self, sales_context: dict) -> str:
        stage = sales_context.get('stage', 'unknown')
        print(f"GenericSalesSkills: Suggesting closing technique for sales stage '{stage}'")
        if sales_context.get("strong_interest_expressed"):
            return "It sounds like we've found a great match for your needs. Would you be comfortable moving forward with the next steps today?"
        return "Based on our conversation, it seems like this aligns well with what you're looking for. What are your thoughts on proceeding?"

class RealEstateKnowledgeService:
    def __init__(self, service_config: dict):
        self.config = service_config

        raw_tf_base_path_val = self.config.get('tf_model_base_path', '/app/ml_models/')
        self.tf_model_base_path_raw_debug = raw_tf_base_path_val # Debug raw value
        self.tf_model_base_path = resolve_config_value(raw_tf_base_path_val, default_if_placeholder_not_set='/app/ml_models/')

        self.property_embedding_model_weights = self.config.get('property_embedding_model_weights')
        self.prospect_embedding_model_weights = self.config.get('prospect_embedding_model_weights')
        self.matching_model_weights = self.config.get('matching_model_weights')

        db_connections_config = self.config.get('database_connections', {})
        mls_db_config = db_connections_config.get('mls_db', {})
        regional_db_config = db_connections_config.get('regional_metrics_db', {})

        self.mls_db_settings = {
            "type": mls_db_config.get('type', 'postgresql'),
            "host": resolve_config_value(mls_db_config.get('host'), "localhost"),
            "port": resolve_config_value(mls_db_config.get('port'), 5432, target_type=int),
            "user": resolve_config_value(mls_db_config.get('user'), ""),
            "password": resolve_config_value(mls_db_config.get('password'), ""),
            "database_name": resolve_config_value(mls_db_config.get('database_name'), "mls_data")
        }

        self.regional_db_path = resolve_config_value(
            regional_db_config.get('path'),
            "/path/to/data/real_estate/regional_metrics.db"
        )

        raw_glossary_path = self.config.get('real_estate_glossary_path')
        self.real_estate_glossary_path = resolve_config_value(
            raw_glossary_path,
            "path/to/data/real_estate/glossary.json"
        )

        raw_legal_docs_path = self.config.get('legal_document_templates_path')
        self.legal_document_templates_path = resolve_config_value(
            raw_legal_docs_path,
            "path/to/data/real_estate/legal_docs/"
        )

        tf_base_path_to_print = getattr(self, 'tf_model_base_path', 'NOT SET')
        regional_db_path_to_print = getattr(self, 'regional_db_path', 'NOT SET')
        glossary_path_to_print = getattr(self, 'real_estate_glossary_path', 'NOT SET')
        legal_docs_path_to_print = getattr(self, 'legal_document_templates_path', 'NOT SET')

        print(f"RealEstateKnowledgeService Initialized (TF Models Base Raw: {self.tf_model_base_path_raw_debug}, "
              f"TF Models Base Resolved: {tf_base_path_to_print}, "
              f"MLS DB: {self.mls_db_settings.get('host')}, Regional DB: {regional_db_path_to_print}, "
              f"Glossary: {glossary_path_to_print}, Legal Docs: {legal_docs_path_to_print})")

    def get_property_details(self, address_or_mls: str) -> dict:
        print(f"RealEstateKnowledge: Getting details for '{address_or_mls}'.")
        return {"address": address_or_mls, "price": 500000, "beds": 3, "baths": 2, "sqft": 1800, "status": "Available", "description": "Charming colonial in a quiet neighborhood."}

    def find_matching_properties(self, prospect_profile: dict, market_context: dict = None, seasonality_context: dict = None) -> list:
        prospect_name = prospect_profile.get('name', 'N/A')
        print(f"RealEstateKnowledge: Finding matches for prospect: {prospect_name}")
        sim_matches = [
            {"address": "123 Elm Street, Springfield", "price": 760000, "beds": 4, "baths": 3, "sqft": 2500, "match_score": 0.92, "reason": "Great location fit and features, slightly above ideal budget but high overall match."},
            {"address": "456 Oak Avenue, Springfield", "price": 720000, "beds": 3, "baths": 2.5, "sqft": 2200, "match_score": 0.88, "reason": "Excellent price fit, good style match, meets all core needs."},
        ]
        return sim_matches

    def get_market_analysis(self, area_criteria: dict) -> dict:
        location = area_criteria.get('location', 'the general area')
        print(f"RealEstateKnowledge: Getting market analysis for area: {location}")
        return {"area": location, "avg_price": 650000, "trend": "stable", "days_on_market": 35, "advice": "A steady market."}


class SalesAgentService:
    def __init__(self, config: dict, generic_sales_service: GenericSalesSkillService, real_estate_service: RealEstateKnowledgeService = None):
        self.service_config = config.get('sales_agent_service', {})
        self.generic_sales = generic_sales_service
        self.real_estate = real_estate_service
        default_stage_raw = self.service_config.get('default_sales_stage', "greeting")
        self.default_sales_stage = resolve_config_value(default_stage_raw, "greeting")
        print(f"SalesAgentService Initialized (Default Stage: {self.default_sales_stage}, Specialist: {'RealEstate' if real_estate_service else 'Generic'})")

    def generate_sales_response(self, sales_context: dict, user_input_details: dict, emotion_data: dict = None) -> str:
        intent = user_input_details.get("intent")
        print(f"SalesAgentService: Handling intent '{intent}' in stage '{sales_context.get('stage')}'")
        if sales_context.get("stage") == "greeting":
            sales_context["stage"] = "discovery_initial"
            return self.generic_sales.get_opening_line(sales_context.get("prospect_profile"))
        return "Sales response based on intent and stage."


if __name__ == '__main__':
    # This sys.path manipulation is for allowing direct execution of the service file
    # if config_utils is in the parent directory.
    if "config_utils" not in sys.modules:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        try:
            from config_utils import resolve_config_value as main_resolve_config_value # Use if available
        except ImportError:
            main_resolve_config_value = resolve_config_value # Fallback to local one for direct run
            print("Warning: Could not import resolve_config_value from config_utils. Using local fallback for __main__.")

    dummy_app_config = {
        "generic_sales_skill_service": {
            "model_path": "${GENERIC_SALES_MODEL_PATH_TEST:-sim/sales_model.pkl}",
            "playbooks": {
                "opening_scripts_path": "${GS_OPENING_PATH_TEST:-sim/openings.csv}",
                "objection_handling_path": "${GS_OBJECTION_PATH_TEST:-sim/objections.json}",
                "closing_techniques_path": "${GS_CLOSING_PATH_TEST:-sim/closings.txt}"
            }
        },
        "real_estate_knowledge_service": {
            "tf_model_base_path": "${RE_TF_MODELS_PATH_TEST:-sim/tf_models/}",
            "database_connections": {
                "regional_metrics_db": {"path": "${RE_REGIONAL_DB_PATH_TEST:-sim/regional.db}"}
            },
            "real_estate_glossary_path": "${RE_GLOSSARY_PATH_TEST:-sim/glossary.json}",
            "legal_document_templates_path": "${RE_LEGAL_DOCS_PATH_TEST:-sim/legal_docs/}"
        },
        "sales_agent_service": { "default_sales_stage": "greeting" }
    }
    os.environ["GS_OPENING_PATH_TEST"] = "env_openings.csv"
    os.environ["RE_REGIONAL_DB_PATH_TEST"] = "env_regional.db"
    os.environ["RE_TF_MODELS_PATH_TEST"] = "env_tf_models/" # Ensure this is set for the __main__ test


    gen_sales_config = dummy_app_config['generic_sales_skill_service']
    # Manually resolve for __main__ if using main_resolve_config_value that might be different
    # gen_sales_config['model_path'] = main_resolve_config_value(gen_sales_config['model_path'])
    # ... and for playbooks ...
    gen_sales = GenericSalesSkillService(service_config=gen_sales_config)
    print(f"Initialized GenericSales with playbooks: {gen_sales.playbooks}")

    re_knowledge_config = dummy_app_config['real_estate_knowledge_service']
    # Manually resolve for __main__
    # re_knowledge_config['tf_model_base_path'] = main_resolve_config_value(re_knowledge_config['tf_model_base_path'])
    # ... and for other paths ...
    re_knowledge = RealEstateKnowledgeService(service_config=re_knowledge_config)

    del os.environ["GS_OPENING_PATH_TEST"]
    del os.environ["RE_REGIONAL_DB_PATH_TEST"]
    del os.environ["RE_TF_MODELS_PATH_TEST"]

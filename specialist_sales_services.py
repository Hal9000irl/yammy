# specialist_sales_services.py
# Contains services for generic sales skills, real estate knowledge, and sales agent orchestration.

import time # For simulation
import os # For os.getenv

# Attempt to import resolve_config_value from main.
try:
    from main import resolve_config_value
except ImportError:
    # Fallback basic version if direct import fails
    def resolve_config_value(value_from_config, default_if_placeholder_not_set=None, target_type=str):
        if isinstance(value_from_config, str) and value_from_config.startswith("${") and value_from_config.endswith("}"):
            var_name = value_from_config.strip("${}")
            val = os.getenv(var_name, default_if_placeholder_not_set)
            # Basic type casting for int, add others if needed by this file
            if target_type == int and val is not None: return int(val)
            return val if target_type == str else None # Keep it simple for fallback
        # Basic type casting for int
        if target_type == int and value_from_config is not None: return int(value_from_config)
        return value_from_config


class GenericSalesSkillService:
    """
    Provides foundational sales skills (openings, objection handling, closing).
    Trained on general sales data.
    """
    def __init__(self, service_config: dict): # Changed to accept service_config directly
        self.config = service_config

        raw_model_path = self.config.get('model_path', 'default_sales_model.pkl')
        self.model_path = resolve_config_value(raw_model_path, default_if_placeholder_not_set='default_sales_model.pkl')

        self.playbooks = self.config.get('playbooks', {}) # Playbook paths might also use resolver if they become env-dependent
        print(f"GenericSalesSkillService Initialized (Model: {self.model_path}, Playbooks: {list(self.playbooks.keys())})")
        # Real: Load model, parse playbooks

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
        elif "not interested" in objection_lower or "no time" in objection_lower:
            return "I appreciate your honesty. Before you go, would you mind if I quickly asked what your current real estate plans are, just so I know if there's a better time to reach out or if our services aren't a fit right now?"
        elif "already working with someone" in objection_lower:
            return "That's great to hear you're already taking steps! Out of curiosity, are you fully committed, or would you be open to a second opinion on how to maximize your outcome?"
        return "That's a valid point. Let me share some information that might provide a different perspective on that..."

    def suggest_closing_technique(self, sales_context: dict) -> str:
        stage = sales_context.get('stage', 'unknown')
        print(f"GenericSalesSkills: Suggesting closing technique for sales stage '{stage}'")
        if sales_context.get("strong_interest_expressed"):
            return "It sounds like we've found a great match for your needs. Would you be comfortable moving forward with the next steps today?"
        return "Based on our conversation, it seems like this aligns well with what you're looking for. What are your thoughts on proceeding?"

class RealEstateKnowledgeService:
    """
    Provides real estate specific knowledge, including property matching.
    Integrates your TensorFlow models and custom algorithms/databases.
    """
    def __init__(self, service_config: dict): # Changed to accept service_config directly
        self.config = service_config

        raw_tf_base_path = self.config.get('tf_model_base_path', './ml_models/')
        self.tf_base_path = resolve_config_value(raw_tf_base_path, default_if_placeholder_not_set='./ml_models/')

        self.database_connections = self.config.get('database_connections', {})
        mls_db_config = self.database_connections.get('mls_db', {})

        self.mls_db_settings = {
            "type": mls_db_config.get('type', 'postgresql'),
            "host": resolve_config_value(mls_db_config.get('host'), "localhost"),
            "port": resolve_config_value(mls_db_config.get('port'), 5432, target_type=int),
            "user": resolve_config_value(mls_db_config.get('user'), ""),
            "password": resolve_config_value(mls_db_config.get('password'), ""),
            "database_name": resolve_config_value(mls_db_config.get('database_name'), "mls_data")
        }
        # Note: For other DBs like regional_metrics_db, similar resolving would be needed if their paths/params are placeholders.

        print(f"RealEstateKnowledgeService Initialized (TF Models Base: {self.tf_base_path}, MLS DB Settings: {self.mls_db_settings})")

    def get_property_details(self, address_or_mls: str) -> dict:
        print(f"RealEstateKnowledge: Getting details for '{address_or_mls}'.")
        return {"address": address_or_mls, "price": 500000, "beds": 3, "baths": 2, "sqft": 1800, "status": "Available", "description": "Charming colonial in a quiet neighborhood."}

    def find_matching_properties(self, prospect_profile: dict, market_context: dict = None, seasonality_context: dict = None) -> list:
        prospect_name = prospect_profile.get('name', 'N/A')
        print(f"RealEstateKnowledge: Finding matches for prospect: {prospect_name}")
        sim_matches = [
            {"address": "123 Elm Street, Springfield", "price": 760000, "beds": 4, "baths": 3, "sqft": 2500, "match_score": 0.92, "reason": "Great location fit and features, slightly above ideal budget but high overall match."},
            {"address": "456 Oak Avenue, Springfield", "price": 720000, "beds": 3, "baths": 2.5, "sqft": 2200, "match_score": 0.88, "reason": "Excellent price fit, good style match, meets all core needs."},
            {"address": "789 Pine Lane, Shelbyville", "price": 680000, "beds": 3, "baths": 2, "sqft": 1900, "match_score": 0.85, "reason": "Very good budget fit, slightly smaller but in a desirable alternative area."}
        ]
        preferred_location = prospect_profile.get("area_interest", "").lower()
        if preferred_location:
            return [m for m in sim_matches if preferred_location in m["address"].lower()] or sim_matches[:1]
        return sim_matches[:2]

    def get_market_analysis(self, area_criteria: dict) -> dict:
        location = area_criteria.get('location', 'the general area')
        print(f"RealEstateKnowledge: Getting market analysis for area: {location}")
        if "downtown" in location.lower():
            return {"area": location, "avg_price": 750000, "trend": "hot seller's market", "days_on_market": 21, "advice": "Properties are moving quickly, so prompt decisions are key."}
        elif "suburban" in location.lower():
            return {"area": location, "avg_price": 620000, "trend": "balanced market", "days_on_market": 45, "advice": "Good inventory available, more room for negotiation."}
        return {"area": location, "avg_price": 650000, "trend": "stable", "days_on_market": 35, "advice": "A steady market with opportunities for both buyers and sellers."}

class SalesAgentService:
    """
    Orchestrates sales interactions, combining generic skills with niche knowledge.
    """
    def __init__(self, config: dict, generic_sales_service: GenericSalesSkillService, real_estate_service: RealEstateKnowledgeService = None):
        self.service_config = config.get('sales_agent_service', {})
        self.generic_sales = generic_sales_service
        self.real_estate = real_estate_service

        default_stage_raw = self.service_config.get('default_sales_stage', "greeting")
        self.default_sales_stage = resolve_config_value(default_stage_raw, "greeting")

        print(f"SalesAgentService Initialized (Default Stage: {self.default_sales_stage}, Specialist: {'RealEstate' if real_estate_service else 'Generic'})")

    def generate_sales_response(self, sales_context: dict, user_input_details: dict, emotion_data: dict = None) -> str:
        intent = user_input_details.get("intent")
        entities = user_input_details.get("entities", {})
        user_text = user_input_details.get("text", "")
        print(f"SalesAgentService: Handling intent '{intent}' (Entities: {entities}) in stage '{sales_context.get('stage')}'")
        response_text = ""

        if "stage" not in sales_context: sales_context["stage"] = self.default_sales_stage
        if "prospect_profile" not in sales_context: sales_context["prospect_profile"] = {}

        if sales_context["stage"] == "greeting":
            response_text = self.generic_sales.get_opening_line(sales_context.get("prospect_profile"))
            sales_context["stage"] = "discovery_initial"
        elif intent == "inquire_real_estate" and self.real_estate:
            location_interest = entities.get("location", "your desired area")
            sales_context["prospect_profile"]["area_interest"] = location_interest
            market_info = self.real_estate.get_market_analysis({"location": location_interest})
            response_text = (f"Certainly! Regarding the real estate market in {market_info.get('area')}, "
                             f"it's currently a {market_info.get('trend')}. The average price is around ${market_info.get('avg_price'):,}, "
                             f"and homes are typically on the market for about {market_info.get('days_on_market')} days. "
                             f"{market_info.get('advice', '')} "
                             f"To help me find the best options for you, could you tell me a bit about what you're looking for in a property?")
            sales_context["stage"] = "needs_assessment_real_estate"
        elif sales_context["stage"] == "needs_assessment_real_estate" and self.real_estate:
            sales_context["prospect_profile"]["beds_desired"] = 3
            sales_context["prospect_profile"]["budget_max"] = 800000
            print(f"SalesAgentService: Prospect profile updated with needs: {sales_context['prospect_profile']}")
            matches = self.real_estate.find_matching_properties(sales_context["prospect_profile"])
            if matches:
                response_text = f"Based on what you've told me, I found a few interesting properties. For example, there's one at {matches[0]['address']} listed at ${matches[0]['price']:,}. It has {matches[0].get('beds','N/A')} beds and {matches[0].get('baths','N/A')} baths. {matches[0].get('reason','')} Would you like to hear more about this one, or perhaps another option?"
                sales_context["current_property_discussion"] = matches[0]
                sales_context["stage"] = "property_presentation"
            else:
                response_text = "I'm searching for properties based on your criteria. While I do that, could you tell me about any must-have features for your new home?"
                sales_context["stage"] = "further_needs_assessment"
        elif intent == "price_objection":
            response_text = self.generic_sales.handle_objection(user_text, sales_context)
            if self.real_estate and sales_context.get("current_property_discussion"):
                response_text += f" Regarding the property at {sales_context['current_property_discussion']['address']}, we could explore if there's any flexibility or look at alternatives that better align with your budget expectations. What range were you considering?"
                sales_context["stage"] = "budget_reassessment"
            else:
                sales_context["stage"] = "objection_response"
        elif intent == "affirmative" and sales_context["stage"] == "property_presentation":
             current_prop = sales_context.get("current_property_discussion")
             if current_prop:
                 response_text = f"Great! The property at {current_prop['address']} also features {current_prop.get('description', 'several nice amenities')}. We could schedule a viewing if you're interested. What are your thoughts?"
                 sales_context["stage"] = "closing_for_viewing"
             else:
                 response_text = "Which property were you interested in hearing more about?"
        else:
            response_text = f"I see. To ensure I'm on the right track, what's the most important factor for you in this decision?"
            sales_context["stage"] = "clarification_sales"

        print(f"SalesAgentService: New sales stage: {sales_context.get('stage')}")
        return response_text

if __name__ == '__main__':
    # Dummy global config for direct execution testing
    dummy_app_config = {
        "generic_sales_skill_service": {
            "model_path": "${GENERIC_SALES_MODEL_PATH_TEST:-sim/sales_model.pkl}",
            "playbooks": {"opening_scripts_path": "sim/openings.csv"}
        },
        "real_estate_knowledge_service": {
            "tf_model_base_path": "${RE_TF_MODELS_PATH_TEST:-sim/tf_models/}",
            "database_connections": {
                "mls_db": {
                    "host": "${MLS_DB_HOST_TEST:-sim_host}",
                    "port": "${MLS_DB_PORT_TEST:-1234}", # Test int conversion
                    "user": "${MLS_DB_USER_TEST:-sim_user}",
                    "password": "${MLS_DB_PASS_TEST:-sim_pass}",
                    "database_name": "${MLS_DB_NAME_TEST:-sim_db}"
                }
            }
        },
        "sales_agent_service": {
            "default_sales_stage": "greeting"
        }
    }

    # Simulate setting some environment variables for testing the resolver
    os.environ["GENERIC_SALES_MODEL_PATH_TEST"] = "env_sales_model.pkl"
    os.environ["MLS_DB_PORT_TEST"] = "5432" # Env vars are strings

    # Pass the specific service config to each service
    gen_sales = GenericSalesSkillService(service_config=dummy_app_config['generic_sales_skill_service'])
    re_knowledge = RealEstateKnowledgeService(service_config=dummy_app_config['real_estate_knowledge_service'])
    # SalesAgentService takes the global config because it might access other parts,
    # but its own settings are within 'sales_agent_service' key.
    sales_agent = SalesAgentService(config=dummy_app_config, generic_sales_service=gen_sales, real_estate_service=re_knowledge)

    context = {"stage": "greeting", "prospect_profile": {}}
    user_details_greet = {"intent": "greet", "text": "Hello"}
    print(f"\nUser: Hello")
    print(f"Agent: {sales_agent.generate_sales_response(context, user_details_greet)}")

    # Clean up env vars
    del os.environ["GENERIC_SALES_MODEL_PATH_TEST"]
    del os.environ["MLS_DB_PORT_TEST"]
# Removed the stray ``` marker from the end of the file.

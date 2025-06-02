# specialist_sales_services.py
# Contains services for generic sales skills, real estate knowledge, and sales agent orchestration.

import time # For simulation

class GenericSalesSkillService:
    """
    Provides foundational sales skills (openings, objection handling, closing).
    Trained on general sales data.
    """
    def __init__(self, config: dict):
        self.config = config.get('generic_sales_skill_service', {})
        self.model_path = self.config.get('model_path', 'default_sales_model.pkl')
        self.playbooks = self.config.get('playbooks', {})
        print(f"GenericSalesSkillService Initialized (Model: {self.model_path}, Playbooks: {list(self.playbooks.keys())})")
        # Real: Load model, parse playbooks

    def get_opening_line(self, prospect_profile: dict = None) -> str:
        print("GenericSalesSkills: Getting opening line.")
        # Real: Use model or playbook logic based on prospect_profile
        if prospect_profile and prospect_profile.get('type') == 'cold_lead':
            return "Hi, this is Alex calling from Premier Properties. I came across your information and wanted to see if you might be considering a move in the near future?"
        return "Thanks for taking my call today. I'm calling from Premier Properties, and I wanted to briefly discuss how we might be able to assist with your real estate needs."

    def handle_objection(self, objection_text: str, sales_context: dict) -> str:
        stage = sales_context.get('stage', 'unknown')
        print(f"GenericSalesSkills: Handling objection '{objection_text}' in sales stage '{stage}'")
        # Real: Use more sophisticated objection handling model/rules
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
        # Real: Based on context, suggest appropriate closing technique
        if sales_context.get("strong_interest_expressed"):
            return "It sounds like we've found a great match for your needs. Would you be comfortable moving forward with the next steps today?"
        return "Based on our conversation, it seems like this aligns well with what you're looking for. What are your thoughts on proceeding?"

class RealEstateKnowledgeService:
    """
    Provides real estate specific knowledge, including property matching.
    Integrates your TensorFlow models and custom algorithms/databases.
    """
    def __init__(self, config: dict):
        self.config = config.get('real_estate_knowledge_service', {})
        self.tf_base_path = self.config.get('tf_model_base_path', './ml_models/')
        # from tf_property_model import PropertyEmbeddingModel # Real: Uncomment and import
        # from tf_prospect_model import ProspectEmbeddingModel
        # from tf_matching_model import MatchingModel
        # self.property_embedder = PropertyEmbeddingModel(weights_path=self.tf_base_path + self.config.get('property_embedding_model_weights'))
        # self.prospect_embedder = ProspectEmbeddingModel(weights_path=self.tf_base_path + self.config.get('prospect_embedding_model_weights'))
        # self.matcher = MatchingModel(weights_path=self.tf_base_path + self.config.get('matching_model_weights'))
        print(f"RealEstateKnowledgeService Initialized (TF Models Base: {self.tf_base_path}, DBs: {list(self.config.get('database_connections', {}).keys())})")
        # Real: Initialize connections to MLS, your custom databases

    def get_property_details(self, address_or_mls: str) -> dict:
        print(f"RealEstateKnowledge: Getting details for '{address_or_mls}'.")
        # Real: Query MLS or property database
        return {"address": address_or_mls, "price": 500000, "beds": 3, "baths": 2, "sqft": 1800, "status": "Available", "description": "Charming colonial in a quiet neighborhood."}

    def find_matching_properties(self, prospect_profile: dict, market_context: dict = None, seasonality_context: dict = None) -> list:
        """Uses the TF embedding and matching models."""
        prospect_name = prospect_profile.get('name', 'N/A')
        print(f"RealEstateKnowledge: Finding matches for prospect: {prospect_name}")
        # 1. Preprocess prospect_profile for self.prospect_embedder
        #    processed_prospect_data = self._preprocess_prospect_for_tf(prospect_profile)
        #    prospect_emb = self.prospect_embedder.predict(processed_prospect_data)
        # 2. Fetch candidate properties from DB/MLS based on broad criteria from prospect_profile
        #    candidate_properties = self._fetch_candidate_properties(prospect_profile)
        # 3. For each candidate property:
        #    processed_property_data = self._preprocess_property_for_tf(property_data)
        #    prop_emb = self.property_embedder.predict(processed_property_data)
        # 4. For each pair:
        #    match_scores = self.matcher.compute_match(prop_emb, prospect_emb, market_context or {}, seasonality_context or {})
        # 5. Rank and return top matches with scores and reasons
        
        # Simulated response:
        sim_matches = [
            {"address": "123 Elm Street, Springfield", "price": 760000, "beds": 4, "baths": 3, "sqft": 2500, "match_score": 0.92, "reason": "Great location fit and features, slightly above ideal budget but high overall match."},
            {"address": "456 Oak Avenue, Springfield", "price": 720000, "beds": 3, "baths": 2.5, "sqft": 2200, "match_score": 0.88, "reason": "Excellent price fit, good style match, meets all core needs."},
            {"address": "789 Pine Lane, Shelbyville", "price": 680000, "beds": 3, "baths": 2, "sqft": 1900, "match_score": 0.85, "reason": "Very good budget fit, slightly smaller but in a desirable alternative area."}
        ]
        # Filter by prospect's preferred location if available
        preferred_location = prospect_profile.get("area_interest", "").lower()
        if preferred_location:
            return [m for m in sim_matches if preferred_location in m["address"].lower()] or sim_matches[:1] # Return filtered or first if no match
        return sim_matches[:2] # Return top 2 by default

    def get_market_analysis(self, area_criteria: dict) -> dict:
        location = area_criteria.get('location', 'the general area')
        print(f"RealEstateKnowledge: Getting market analysis for area: {location}")
        # Real: Query your regional_metrics_db
        if "downtown" in location.lower():
            return {"area": location, "avg_price": 750000, "trend": "hot seller's market", "days_on_market": 21, "advice": "Properties are moving quickly, so prompt decisions are key."}
        elif "suburban" in location.lower():
            return {"area": location, "avg_price": 620000, "trend": "balanced market", "days_on_market": 45, "advice": "Good inventory available, more room for negotiation."}
        return {"area": location, "avg_price": 650000, "trend": "stable", "days_on_market": 35, "advice": "A steady market with opportunities for both buyers and sellers."}

class SalesAgentService:
    """
    Orchestrates sales interactions, combining generic skills with niche knowledge.
    This is the primary specialist called by Rasa for sales-related intents.
    """
    def __init__(self, config: dict, generic_sales_service: GenericSalesSkillService, real_estate_service: RealEstateKnowledgeService = None):
        self.config = config.get('sales_agent_service', {})
        self.generic_sales = generic_sales_service
        self.real_estate = real_estate_service # Injected for specific niches like real estate
        print(f"SalesAgentService Initialized (Default Stage: {self.config.get('default_sales_stage')}, Specialist: {'RealEstate' if real_estate_service else 'Generic'})")

    def generate_sales_response(self, sales_context: dict, user_input_details: dict, emotion_data: dict = None) -> str:
        """
        Generates a sales-focused response.
        Manages and updates the sales_context.
        """
        intent = user_input_details.get("intent")
        entities = user_input_details.get("entities", {})
        user_text = user_input_details.get("text", "")

        print(f"SalesAgentService: Handling intent '{intent}' (Entities: {entities}) in stage '{sales_context.get('stage')}'")
        response_text = ""

        # Initialize stage if not present
        if "stage" not in sales_context:
            sales_context["stage"] = self.config.get('default_sales_stage', "greeting")
        if "prospect_profile" not in sales_context:
            sales_context["prospect_profile"] = {}


        # Logic flow based on intent and sales stage
        if sales_context["stage"] == "greeting":
            response_text = self.generic_sales.get_opening_line(sales_context.get("prospect_profile"))
            sales_context["stage"] = "discovery_initial"
        
        elif intent == "inquire_real_estate" and self.real_estate:
            location_interest = entities.get("location", "your desired area")
            sales_context["prospect_profile"]["area_interest"] = location_interest # Store interest
            
            market_info = self.real_estate.get_market_analysis({"location": location_interest})
            response_text = (f"Certainly! Regarding the real estate market in {market_info.get('area')}, "
                             f"it's currently a {market_info.get('trend')}. The average price is around ${market_info.get('avg_price'):,}, "
                             f"and homes are typically on the market for about {market_info.get('days_on_market')} days. "
                             f"{market_info.get('advice', '')} "
                             f"To help me find the best options for you, could you tell me a bit about what you're looking for in a property?")
            sales_context["stage"] = "needs_assessment_real_estate"

        elif sales_context["stage"] == "needs_assessment_real_estate" and self.real_estate:
            # Here you'd parse user_text for needs (beds, baths, budget, etc.)
            # For simulation, assume we got some criteria
            sales_context["prospect_profile"]["beds_desired"] = 3 # Simulated
            sales_context["prospect_profile"]["budget_max"] = 800000 # Simulated
            print(f"SalesAgentService: Prospect profile updated with needs: {sales_context['prospect_profile']}")
            
            matches = self.real_estate.find_matching_properties(sales_context["prospect_profile"])
            if matches:
                response_text = f"Based on what you've told me, I found a few interesting properties. For example, there's one at {matches[0]['address']} listed at ${matches[0]['price']:,}. It has {matches[0].get('beds','N/A')} beds and {matches[0].get('baths','N/A')} baths. {matches[0].get('reason','')} Would you like to hear more about this one, or perhaps another option?"
                sales_context["current_property_discussion"] = matches[0] # Store for follow-up
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
        
        elif intent == "affirmative" and sales_context["stage"] == "property_presentation": # e.g. user says "yes, tell me more"
             current_prop = sales_context.get("current_property_discussion")
             if current_prop:
                 response_text = f"Great! The property at {current_prop['address']} also features {current_prop.get('description', 'several nice amenities')}. We could schedule a viewing if you're interested. What are your thoughts?"
                 sales_context["stage"] = "closing_for_viewing"
             else:
                 response_text = "Which property were you interested in hearing more about?"


        else: # Fallback sales response
            response_text = f"I see. To ensure I'm on the right track, what's the most important factor for you in this decision?"
            sales_context["stage"] = "clarification_sales"

        print(f"SalesAgentService: New sales stage: {sales_context.get('stage')}")
        return response_text

if __name__ == '__main__':
    dummy_config = {
        "generic_sales_skill_service": {
            "playbooks": {"opening_scripts_path": "sim/openings.csv"}
        },
        "real_estate_knowledge_service": {
            "tf_model_base_path": "sim/tf_models/",
            "database_connections": {"mls_db": {"host": "sim_host"}}
        },
        "sales_agent_service": {
            "default_sales_stage": "greeting"
        }
    }
    gen_sales = GenericSalesSkillService(config=dummy_config)
    re_knowledge = RealEstateKnowledgeService(config=dummy_config)
    sales_agent = SalesAgentService(config=dummy_config, generic_sales_service=gen_sales, real_estate_service=re_knowledge)

    # Test SalesAgentService
    context = {"stage": "greeting", "prospect_profile": {}}
    user_details_greet = {"intent": "greet", "text": "Hello"}
    print(f"\nUser: Hello")
    print(f"Agent: {sales_agent.generate_sales_response(context, user_details_greet)}")
    
    user_details_inquire = {"intent": "inquire_real_estate", "entities": {"location": "downtown"}, "text": "I want to sell my house in downtown"}
    print(f"\nUser: I want to sell my house in downtown")
    print(f"Agent: {sales_agent.generate_sales_response(context, user_details_inquire)}")

    # Assume user provided some needs, now in 'needs_assessment_real_estate' stage
    print(f"\nUser: I need 3 bedrooms and a budget of $700k (Simulated context update)")
    context['stage'] = 'needs_assessment_real_estate' # Manually update stage for simulation
    context['prospect_profile']['area_interest'] = 'downtown' # From previous turn
    user_details_needs = {"intent": "provide_info", "text": "I need 3 bedrooms and a budget of $700k"} # Rasa would parse this
    print(f"Agent: {sales_agent.generate_sales_response(context, user_details_needs)}")
    
    user_details_objection = {"intent": "price_objection", "text": "That's too expensive"}
    print(f"\nUser: That's too expensive")
    print(f"Agent: {sales_agent.generate_sales_response(context, user_details_objection)}")

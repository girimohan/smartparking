import json
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionCheckParkingAvailability(Action):
    def name(self) -> Text:
        return "action_check_parking_availability"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        location = next(tracker.get_latest_entity_values("location"), None)
        
        if not location:
            dispatcher.utter_message(text="I'm sorry, I couldn't determine the location. Can you please specify where you're looking for parking?")
            return []

        try:
            with open('parking_data.json', 'r') as f:
                parking_data = json.load(f)
            
            if location.lower() in parking_data:
                available_spaces = parking_data[location.lower()]
                response = f"In {location}, there are currently {available_spaces} parking spaces available."
            else:
                response = f"I'm sorry, I don't have information about parking in {location}. The locations I have data for are: {', '.join(parking_data.keys())}."
        except FileNotFoundError:
            response = "I'm sorry, I don't have any parking information available at the moment. Please try again later."
        except json.JSONDecodeError:
            response = "I'm sorry, there was an error reading the parking information. Please try again later."
        
        dispatcher.utter_message(text=response)
        
        return []
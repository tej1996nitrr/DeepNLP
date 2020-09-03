from rasa_sdk import Action
from rasa_sdk.events import SlotSet
import requests
import json
import pandas as pd
class ActionWeather(Action):
    def name(self):
        return 'action_weather'

    def run(self, dispatcher, tracker, domain):
        from apixu.client import ApixuClient
        api_key = 'a7a24285bb778fd47adcf793e0495371'  # your apixu key
        client = ApixuClient(api_key)
        loc = tracker.get_slot('location')
        url = f"http://api.weatherstack.com/current?access_key={api_key}&query={loc}"
        weather_requests = requests.get(url)
        json_data = weather_requests.json()
        country = json_data['location']['country']
        city = json_data['location']['name']
        condition = json_data['current']['weather_descriptions'][0]
        temperature_c = json_data['current']['temperature']
        humidity = json_data['current']['humidity']
        wind_mph = json_data['current']['wind_speed']

        response = """It is currently {} in {} at the moment. The temperature is {} degrees, the humidity is {}% and the wind speed is {} mph.""".format(
            condition, city, temperature_c, humidity, wind_mph)
        print(response)
        dispatcher.utter_message(response)
        return [SlotSet('location', loc)]
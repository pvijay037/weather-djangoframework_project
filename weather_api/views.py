# weather_app/views.py
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from sklearn.model_selection import train_test_split
from .ml_model import WeatherPredictionModel  # WeatherPredictionModel class defined in ml_model.py
class WeatherAPIView(APIView):
    def get(self, request):
        city = request.query_params.get('city')
        api_key = 'd066a10295f2be5c39b1352aec2c25d5'
        
        # Fetch current weather data
        url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
        response = requests.get(url)
        data = response.json()
        print(data,'hhhh')
        name = data['name']
        lon = data['coord']['lon']
        lat = data['coord']['lat']
        temp = round(data['main']['temp']- 273.15,2)
        humidity = data['main']['humidity']
        country = data['sys']['country']
        dt = data['dt']
        weather=data['weather'][0]['description']

        
        # add more data points from the response as needed
        
        current_weather_data = {
            'city': name,
            'longitude': lon,
            'latitude': lat,
           'datetime':dt,
            'temperature': temp,
            'humidity': humidity,
            'description': weather,
            'country':country,
        }
        
        return Response(current_weather_data)

class EnhancedWeatherPrediction(APIView):
    def get(self, request):
        city = request.query_params.get('city')  # Get the city from query parameter
        cnt = request.query_params.get('cnt', 5)
        api_key = 'd066a10295f2be5c39b1352aec2c25d5'
        
        # Fetch historical weather data
        url3 = f'http://api.openweathermap.org/data/2.5/forecast?q={city}&cnt={cnt}&appid={api_key}'
        response = requests.get(url3)
        data2 = response.json()
        historical_list = data2['list']
       
        
        historical_weather_data = []
        for historical_entry in historical_list:
            weather_entry = {
                'datetime': historical_entry['dt_txt'],
                'temperature': round(historical_entry['main']['temp'] - 273.15, 2),
                'humidity': historical_entry['main']['humidity'],
                'description': historical_entry['weather'][0]['description'],
            }
            historical_weather_data.append(weather_entry)
        
        # Preprocess historical data and train ML model
       
        historical_features = []
        historical_targets = []
        for entry in historical_weather_data:
            feature_vector = [entry['humidity']]  # Using only humidity as a feature for demonstration
            target = entry['temperature']
            
            historical_features.append(feature_vector)
            historical_targets.append(target)

        X_train, X_test, y_train, y_test = train_test_split(historical_features, historical_targets, test_size=0.2, random_state=42)

        weather_model = WeatherPredictionModel(n_estimators=100, random_state=42)
        weather_model.train(X_train, y_train)

        # Make predictions using the trained model
        predictions = weather_model.predict(X_test)

        # Evaluate the model's performance
        rmse = weather_model.evaluate(X_test, y_test)

        return Response([historical_weather_data,f'Root Mean Squared Error{rmse}'])


# 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

# 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

# d066a10295f2be5c39b1352aec2c25d5


from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import requests

class ChatWeatherView(APIView):
    def fetch_weather(self, city):
        weather_api_key = "d066a10295f2be5c39b1352aec2c25d5"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            weather_description = data['weather'][0]['description']
            temperature = data['main']['temp']
            return f"The weather in {city} is currently {weather_description} with a temperature of {temperature}°C."
        else:
            return "Sorry, I couldn't retrieve the weather information at the moment."


    def generate_response(self, user_input):
        lst=[]
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = TFGPT2LMHeadModel.from_pretrained(model_name)

        conversation = f"User: {user_input}\nBot:"

        input_ids = tokenizer.encode(conversation, return_tensors="tf")

        output = model.generate(input_ids, max_length=150, num_return_sequences=1)

        bot_response = tokenizer.decode(output.numpy()[0], skip_special_tokens=True)
        self.latest_bot_response = bot_response  # Store the latest bot response
        messages = self.latest_bot_response.split("\n")  # Split the bot response by newline
        print(messages, 'hhhhhhhh')
        last_message = messages[-1]
        return last_message
    def post(self, request, *args, **kwargs):
        user_input = request.data.get('user_input', '')

        if user_input.lower() in ['exit', 'quit']:
            bot_response = "Goodbye!"
        else:
            bot_response = self.generate_response(user_input)

            if "" in bot_response.lower():
                city = user_input.split("weather in")[-1].strip()
                weather_info = self.fetch_weather(city)
                bot_response += " " + weather_info

        return Response({"bot_response": bot_response})

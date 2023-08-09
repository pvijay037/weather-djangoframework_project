"""
URL configuration for WeatherForecastBackend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include

# from weather_api.views import WeatherForecastView
from weather_api.views import WeatherAPIView, EnhancedWeatherPrediction, ChatWeatherView
urls_api=[

   path('api/weather/', WeatherAPIView.as_view(), name='weather-api'),
    path('api/enhanced-weather/', EnhancedWeatherPrediction.as_view(), name='enhanced-weather-api'),
    path('api/chat/', ChatWeatherView.as_view(), name='chat'),
]

urlpatterns = [
    path('admin/', admin.site.urls),
    path("",include(urls_api)),



]

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class WeatherPredictionModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.scaler = StandardScaler()  # Standardize features
        
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)  # Scale the features
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
         # Scale the input data
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        mse = mean_squared_error(y, predictions)
        rmse = mse ** 0.5
        return rmse
    




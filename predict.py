import mlflow.pyfunc
import pandas as pd
from pathlib import Path

# 1. Locate the best model
# We point to the local mlruns folder we created
tracking_uri = Path("outputs/mlruns").absolute().as_uri()
mlflow.set_tracking_uri(tracking_uri)

# We'll load the model using the model name we gave it
model_name = "model"
# This fetches the model from the most recent 'best_model' run
model_uri = f"runs:/best_model/{model_name}"

print(f"Loading best model...")
model = mlflow.pyfunc.load_model(model_uri)

# 2. Define a 'Mystery House' 
# Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
mystery_house = pd.DataFrame([{
    "MedInc": 8.32,       # High income area ($83,000+)
    "HouseAge": 41.0,     # Older house (41 years)
    "AveRooms": 6.98,     # ~7 rooms
    "AveBedrms": 1.02,
    "Population": 322.0,  # Small neighborhood
    "AveOccup": 2.55,     # ~2-3 people per house
    "Latitude": 37.88,    # Near Berkeley, CA
    "Longitude": -122.23
}])

# 3. Predict the price!
prediction = model.predict(mystery_house)

print("\n" + "="*40)
print(f"🏠 HOUSE PRICE PREDICTION")
print(f"Estimated Value: ${prediction[0] * 100_000:,.2f}")
print("="*40)
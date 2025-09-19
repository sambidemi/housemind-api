from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib 
import os 
import requests
from fastapi.middleware.cors import CORSMiddleware #needed for managing the frontends that can access the api


def download_from_drive(file_id, filename):
    ## download the file from Google drive if not exists locally
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(url, allow_redirects = True)
        with open(filename, 'wb') as f:
            f.write(r.content)

# downloading the models from google drive
download_from_drive('1APKrwPyT0jGhR5Gq6BVLGDFbzHa4GUle', 'rf_price_prediction.pkl')
download_from_drive('1r2I_MSlKzDD5MP1LYgHxoAtHU35fobjF', 'ROI_predictor.pkl')

# loading the models and encoders into objects

price_model = joblib.load('random_forest_price_prediction.pkl')
city_enc = joblib.load('city_encoder.pkl')
state_enc = joblib.load('state_encoder.pkl')
investment_model = joblib.load('investment_label_model.pkl')
roi_model = joblib.load('ROI_predictor.pkl')

# creating an instance of fastapi
app = FastAPI(title = 'Housemind valuation API', version="1.0.0", description = 'API for communicating with models that performs residential real estate valuation and investment insights.')

# adding corsmiddle
app.add_middleware(CORSMiddleware, 
allow_origins = ['*'],
allow_credentials = True,
allow_headers = ['*'],
allow_methods = ['GET','POST']
)

# defining the input schema for the data input
class pricehousefeatures (BaseModel):
    bed: int = Field(..., example = 3)
    bath: int = Field(..., example = 2)
    acre_lot: float = Field(..., example = 0.12)
    city: str = Field (..., example = 'San Diego')
    state: str = Field (..., example = 'Texas')
    zip_code: int = Field (..., example = 77096)
    house_size: float = Field (..., example = 2500)

# creating a price predict route 
@app.post('/predict')
def predict_price(features : pricehousefeatures):
    try:  
        X = {'bed' : features.bed,
            'bath': features.bath,
            'acre_lot': features.acre_lot,
            'city' : features.city,
            'state' : features.state,
            'zip_code': features.zip_code, 
            'house_size': features.house_size}

        X_test = pd.DataFrame (X, index = [0])
        X_test['city'] = city_enc.transform(X_test['city'])
        X_test['state'] = state_enc.transform (X_test['state'])

        predicted_price = price_model.predict(X_test)
        predicted_price = float(np.round(predicted_price[0], -3)) # we return the predicted price as float so JSON is serializable
        
        return {'Predicted_price' : predicted_price}

    except Exception as e:
        raise HTTPException (status_code = 500, detail = str(e))

# creating an investment prediction route
@app.post('/invest')
def predict_invest(features : pricehousefeatures):
    try: 
        X = {'bed' : features.bed,
            'bath': features.bath,
            'acre_lot': features.acre_lot,
            'city' : features.city,
            'state' : features.state,
            'zip_code': features.zip_code, 
            'house_size': features.house_size}

        X_test = pd.DataFrame (X, index = [0])
        X_test['city'] = city_enc.transform(X_test['city'])
        X_test['state'] = state_enc.transform (X_test['state'])

        predicted_price = price_model.predict(X_test)
        predicted_price = float(np.round(predicted_price[0], -3)) # this rounds the predicted price to the nearest thousands

        X_test['price'] = predicted_price
        # merging the features dataframe for investment model

        X_test = X_test[['price', 'bed', 'bath','acre_lot','city', 'state', 'house_size']]
        investment_label = investment_model.predict(X_test) #predicting the investment label
        
        # merging the features dataframe for ROI model 
        X_test['investment_label'] = investment_label
        roi_score = roi_model.predict(X_test)
        roi_score = float(np.round(roi_score[0],1))
        
        #converting the investment label into strings 
        if investment_label == 0:
            label = 'Market Priced'
        else:
            label = 'Undervalued Opportunity'

        return {'Investment_label' : label, 'ROI_score' : roi_score}
    except Exception as e:
        raise HTTPException (status_code = 500, detail = str(e))

# schema for market_request
class staterequest (BaseModel):
    state : str = Field(..., example = 'Texas')


@app.post('/market_comparison')
def market_comparison(req: staterequest):
    try:
        # load the market data
        state = req.state
        market_data = pd.read_csv('market_comparison_data.csv')
        if state not in market_data['state'].values:
            raise HTTPException (status_code = 404, detaail = 'state is not found')
        state_data = market_data[market_data['state'] == state]
        #total number of houses in 20 cities of that particular state.
        total_number_of_houses = len(state_data)

        #selecting 7 popular cities of that state
        state_city = state_data['city'].value_counts().head(7).index.to_list()
        state_data = state_data[state_data['city'].isin(state_city)]

        #average house price per city
        avg_price_per_city = np.round(state_data.groupby('city')['price'].mean(), -3).to_dict()
        #average roi score per city
        roi_price_per_city = np.round(state_data.groupby('city')['ROI'].mean(), 2).to_dict()

        #distribution of market tiers
        market_tier_distribution = state_data['market_tiers'].value_counts().to_dict()

        #distribution of investment labels
        investment_label_distribution = state_data['investment_label'].value_counts().to_dict()
        return {'total_no_of_houses' : total_number_of_houses, 
        'average_price': avg_price_per_city, 
        'average_roi': roi_price_per_city,
        'market_tier_no': market_tier_distribution,
        'investment_label_no': investment_label_distribution}
    except Exception as e:

        raise HTTPException( status_code = 500, details = str(e))

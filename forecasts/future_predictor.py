# future_predictor.py

import pandas as pd
from features.feature_engineering import create_time_features, create_lag_features

def generate_future_features(df, periods: int = 7):
    future_dates = pd.date_range(start=df['date'].max(), periods=periods + 1, freq='D')[1:]
    future_df = pd.DataFrame({'date': future_dates})
    
    df = pd.concat([df, future_df], ignore_index=True)
    df = create_time_features(df)
    df = create_lag_features(df)

    return df.tail(periods)

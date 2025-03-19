from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
class SubUrbMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, suburb_mean_price):
        self.suburb_mean_price = suburb_mean_price
    
    def fit(self, X, y=None):
        if 'Suburb' not in X.columns:
            raise KeyError("The 'Suburb' column is missing from the input data.")
        return self
    
    def transform(self, X):
        if 'Suburb' not in X.columns:
            raise KeyError("The 'Suburb' column is missing from the data during transformation.")
        
        # Create a copy of the DataFrame to avoid modifying it in place
        X_transformed = X.copy()
        
        # Map the suburb mean price to the 'Suburb' column
        X_transformed['Suburb_Mean_Price'] = X_transformed['Suburb'].map(self.suburb_mean_price)
        X_transformed.drop(columns=['Suburb'], inplace=True)
        return X_transformed

# Custom transformer to handle the Date column conversion and feature extraction
class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting required, just return self
        return self

    def transform(self, X):
        # Ensure that X is a pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # Convert 'Date' column to datetime
            X['Date'] = pd.to_datetime(X['Date'], errors='coerce', dayfirst=True)
            
            # Extract additional features from the 'Date' column
            X['Year'] = X['Date'].dt.year
            X['Month'] = X['Date'].dt.month
            X['Day'] = X['Date'].dt.day
            
            # Drop the original 'Date' column as we now have the extracted features
            X = X.drop(columns=['Date'])
        else:
            raise ValueError("Input must be a pandas DataFrame")
        
        return X
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# --- configuration keys ---
TARGET = "Churn"
LEAKY_ID = "customerID"

def clean(df):
    """Fix the three things wrong with the raw Telco export.

    1. Drop customerID: it is a unique key, so it can only memorize rows (leakage).
    2. TotalCharges arrives as text with blank strings for brand-new customers; coerce it
       to a number so the blanks become NaN for the imputer to fill.
    3. Turn Churn from 'Yes'/'No' into 1/0 so it is a usable label.
    """
    df = df.copy()
    if LEAKY_ID in df.columns:
        df = df.drop(columns=[LEAKY_ID])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load_data(path):
    """Read the CSV and clean it in one call."""
    return clean(pd.read_csv(path))


def feature_columns(df):
    """Split feature columns into numeric and categorical (the target is neither).

    Categorical is defined as "not numeric" rather than "dtype is object", because recent
    pandas stores text as a string dtype, not object, and the object check would miss it.
    """
    numeric = [c for c in df.columns
               if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns
                   if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]
    return numeric, categorical

class TensorPairs(Dataset):
    """Wrap features X and labels y so a DataLoader can batch and shuffle them."""

    def __init__(self, X, y):
        assert len(X) == len(y), "X and y must have the same number of rows"
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loader(X, y, batch_size=64, shuffle=True):
    """Return a DataLoader that yields (batch_X, batch_y) tuples."""
    return DataLoader(TensorPairs(X, y), batch_size=batch_size, shuffle=shuffle)

def prepare_pytorch_tensors(df, numeric_cols, categorical_cols):
    """Process pandas dataframe into numeric matrix and labels ready for PyTorch.
    
    1. Handle missing values (e.g. NaNs in TotalCharges) with simple imputation.
    2. Convert categorical text columns into numerical one-hot vectors.
    """
    # Create copies to prevent modifying the original dataframe
    df_features = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].to_numpy()
    
    # Fill remaining NaNs (like those created in TotalCharges) with the median
    for col in numeric_cols:
        if col in df_features.columns:
            df_features[col] = df_features[col].fillna(df_features[col].median())
            
    # One-hot encode string/categorical columns so they become model-ready floats
    if categorical_cols:
        df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
        
    # Convert features to a clean, single-type numpy matrix for PyTorch ingestion
    X = df_features.to_numpy(dtype="float32")
    
    return X, y

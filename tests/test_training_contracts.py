import pandas as pd
import pytest
from src.models.training import train_models
from src.contracts.training import TrainingResults

@pytest.fixture
def small_synthetic_data():
    # Create minimal data to run training without heavy computation
    data = {
        'Feature1': range(20),
        'Feature2': range(20, 40),
        'Target': range(40, 60)
    }
    df = pd.DataFrame(data)
    
    X = df[['Feature1', 'Feature2']]
    y = df['Target']
    
    return X, y

def test_train_models_contract(small_synthetic_data):
    X, y = small_synthetic_data
    
    # Create dummy scaled data and baseline data
    X_scaled = X.copy()
    Xb = pd.DataFrame({'BaselineFeature': range(20)})
    yb = y.copy()
    
    # Split data into train/test
    X_train, X_test = X.iloc[:15], X.iloc[15:]
    y_train, y_test = y.iloc[:15], y.iloc[15:]
    X_train_scaled, X_test_scaled = X_scaled.iloc[:15], X_scaled.iloc[15:]
    Xb_train_scaled, Xb_test_scaled = Xb.iloc[:15], Xb.iloc[15:]
    yb_train, yb_test = yb.iloc[:15], yb.iloc[15:]

    # Run the training function
    results = train_models(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled,
        Xb_train_scaled, Xb_test_scaled,
        yb_train, yb_test
    )

    # Validate the output contract
    assert isinstance(results, TrainingResults)
    assert hasattr(results, 'baseline')
    assert hasattr(results, 'linear_regression')
    assert hasattr(results, 'random_forest')
    assert hasattr(results, 'gradient_boosting')
    assert hasattr(results, 'rf_model')
    assert hasattr(results, 'gb_model')
    
    # Check that metrics are accessible
    assert results.random_forest.metrics.r2 is not None
    assert results.gradient_boosting.metrics.mae is not None

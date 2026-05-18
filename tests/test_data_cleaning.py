import pandas as pd
import pytest
from src.data.cleaning import clean_data

@pytest.fixture
def raw_data():
    data = {
        'Event': ['A', 'B', 'C', 'D'],
        'Equipment': ['Raw', 'Single-ply', 'Wraps', 'Multi-ply'],
        'TotalKg': [100, 200, None, 400],
        'Tested': ['Yes', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)

def test_clean_data(raw_data):
    cleaned_df = clean_data(raw_data)
    
    # Should drop 'Multi-ply' and 'Single-ply'
    assert 'Multi-ply' not in cleaned_df['Equipment'].values
    assert 'Single-ply' not in cleaned_df['Equipment'].values
    
    # Should drop rows with missing TotalKg
    assert not cleaned_df['TotalKg'].isnull().any()
    
    # Should keep 'Raw' and 'Wraps'
    assert 'Raw' in cleaned_df['Equipment'].values
    
    # Final shape check
    assert len(cleaned_df) == 1
    assert list(cleaned_df.columns) == ['Event', 'Equipment', 'TotalKg', 'Tested']

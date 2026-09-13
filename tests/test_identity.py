import unittest
import pandas as pd
from src.data.identity import build_athlete_id

class TestAthleteIdentity(unittest.TestCase):

    def test_build_athlete_id(self):
        data = {
            'Name': ['John Doe', 'Jane Smith', 'John Doe'],
            'Sex': ['M', 'F', 'M'],
            'WeightClassKg': [85, 63, 85]
        }
        df = pd.DataFrame(data)
        expected_ids = pd.Series(['John Doe_M_85', 'Jane Smith_F_63', 'John Doe_M_85'])
        pd.testing.assert_series_equal(build_athlete_id(df), expected_ids)

    def test_build_athlete_id_missing_columns(self):
        data = {
            'Name': ['John Doe'],
            'Sex': ['M']
        }
        df = pd.DataFrame(data)
        with self.assertRaises(ValueError):
            build_athlete_id(df)

if __name__ == '__main__':
    unittest.main()
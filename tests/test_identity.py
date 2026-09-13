import unittest
import pandas as pd
from src.data.identity import build_athlete_id

class TestAthleteIdentity(unittest.TestCase):

    def test_build_athlete_id_acceptable_strategies(self):
        # Test with minimum acceptable fallback
        data1 = {'Name': ['A'], 'Sex': ['M'], 'WeightClassKg': [85]}
        df1 = pd.DataFrame(data1)
        self.assertEqual(build_athlete_id(df1).iloc[0], 'A_M_85')

        # Test with stronger fallback
        data2 = {'Name': ['A'], 'Sex': ['M'], 'WeightClassKg': [85], 'AgeClass': ['20-23']}
        df2 = pd.DataFrame(data2)
        self.assertEqual(build_athlete_id(df2).iloc[0], 'A_M_85_20-23')
        
    def test_build_athlete_id_with_true_id(self):
        data = {'AthleteID': [123], 'Name': ['A']}
        df = pd.DataFrame(data)
        self.assertEqual(build_athlete_id(df).iloc[0], '123')

    def test_build_athlete_id_fails_with_weak_fallbacks(self):
        # Test that it fails with only Name + Sex
        df_weak1 = pd.DataFrame({'Name': ['A'], 'Sex': ['M']})
        with self.assertRaises(ValueError):
            build_athlete_id(df_weak1)
            
        # Test that it fails with only Name
        df_weak2 = pd.DataFrame({'Name': ['A']})
        with self.assertRaises(ValueError):
            build_athlete_id(df_weak2)

    def test_build_athlete_id_missing_all_columns(self):
        df = pd.DataFrame({'SomeOtherCol': [1]})
        with self.assertRaises(ValueError):
            build_athlete_id(df)
            
    def test_build_athlete_id_with_nans(self):
        data = {'Name': ['A', 'B'], 'Sex': ['M', None], 'WeightClassKg': [85, 85]}
        df = pd.DataFrame(data)
        ids = build_athlete_id(df)
        self.assertEqual(ids.iloc[0], 'A_M_85')
        self.assertEqual(ids.iloc[1], 'B_Unknown_85')

if __name__ == '__main__':
    unittest.main()
import unittest
import pandas as pd
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.features.temporal import feature_engineering
from src.pipelines.walk_forward_pipeline import create_walk_forward_target
from src.models.baselines import PersistenceBaseline

class TestWalkForward(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe that spans several years
        data = {
            'Name': ['Athlete A'] * 10,
            'Sex': ['M'] * 10,
            'WeightClassKg': [85] * 10,
            'Date': pd.to_datetime([f'{2010+i}-01-01' for i in range(10)]),
            'TotalKg': [100 + i * 10 for i in range(10)],
            'Age': [25 + i for i in range(10)],
            'Squat1Kg': [50 + i * 5 for i in range(10)],
            'Squat2Kg': [55 + i * 5 for i in range(10)],
            'Squat3Kg': [60 + i * 5 for i in range(10)],
            'Bench1Kg': [30 + i * 2 for i in range(10)],
            'Bench2Kg': [35 + i * 2 for i in range(10)],
            'Bench3Kg': [40 + i * 2 for i in range(10)],
            'Deadlift1Kg': [60 + i * 5 for i in range(10)],
            'Deadlift2Kg': [65 + i * 5 for i in range(10)],
            'Deadlift3Kg': [70 + i * 5 for i in range(10)],
        }
        self.df_raw = pd.DataFrame(data)
        self.feature_cols = ['Prev_Total'] # Simplified for testing

    def test_min_train_periods(self):
        # Test that evaluation doesn't run if there are not enough periods
        evaluator = WalkForwardEvaluator(models={}, baselines={}, min_train_periods=10)
        evaluator.evaluate(self.df_raw, feature_engineering, create_walk_forward_target, self.feature_cols)
        self.assertEqual(len(evaluator.metrics_results), 0)

        # Test that evaluation runs with sufficient periods
        evaluator = WalkForwardEvaluator(models={'Persistence': PersistenceBaseline()}, baselines={}, min_train_periods=3)
        evaluator.evaluate(self.df_raw, feature_engineering, create_walk_forward_target, self.feature_cols)
        # 10 years total -> 10 - 3 = 7 evaluation windows
        self.assertGreater(len(evaluator.metrics_results), 0)
        self.assertEqual(len(evaluator.metrics_results), 7)

    def test_no_future_leakage_in_folds(self):
        # This is harder to test directly, but we can check that the features for a given test year
        # are based on data from previous years only.
        
        # Let's manually check one fold
        train_df = self.df_raw[self.df_raw['Date'].dt.year < 2015]
        test_df_context = self.df_raw[self.df_raw['Date'].dt.year <= 2015]

        train_engineered = feature_engineering(train_df, athlete_col="Athlete_ID")
        test_engineered = feature_engineering(test_df_context, athlete_col="Athlete_ID")
        
        # The 'Prev_Total' for 2015 should be the 'TotalKg' from 2014
        prev_total_2015 = test_engineered[test_engineered['Date'].dt.year == 2015]['Prev_Total'].iloc[0]
        total_2014 = self.df_raw[self.df_raw['Date'].dt.year == 2014]['TotalKg'].iloc[0]
        
        self.assertEqual(prev_total_2015, total_2014)

if __name__ == '__main__':
    unittest.main()
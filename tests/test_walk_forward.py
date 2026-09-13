import unittest
import pandas as pd
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.features.temporal import feature_engineering
from src.pipelines.walk_forward_pipeline import create_walk_forward_target
from src.models.baselines import PersistenceBaseline
from src.data.identity import build_athlete_id

class TestWalkForward(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe that spans several years with enough data 
        # to survive stricter feature engineering (min_periods=3) and 
        # min_train_periods=5 without collapsing to empty sets.
        # We need more history and more than one event per year for the target to be non-NaN.
        
        years = 15
        
        # Athlete A (15 years of data, 2 events per year)
        data_a = {
            'Name': ['Athlete A'] * years * 2,
            'Sex': ['M'] * years * 2,
            'WeightClassKg': [85] * years * 2,
            'Date': pd.to_datetime([f'{2005+i}-01-01' for i in range(years)] + [f'{2005+i}-07-01' for i in range(years)]),
            'TotalKg': [100 + i * 10 for i in range(years)] + [105 + i * 10 for i in range(years)],
            'Age': [20 + i for i in range(years)] + [20 + i for i in range(years)],
        }
        
        # Athlete B (15 years of data, 2 events per year)
        data_b = {
            'Name': ['Athlete B'] * years * 2,
            'Sex': ['F'] * years * 2,
            'WeightClassKg': [63] * years * 2,
            'Date': pd.to_datetime([f'{2005+i}-03-01' for i in range(years)] + [f'{2005+i}-09-01' for i in range(years)]),
            'TotalKg': [150 + i * 5 for i in range(years)] + [152 + i * 5 for i in range(years)],
            'Age': [22 + i for i in range(years)] + [22 + i for i in range(years)],
        }
        
        df_a = pd.DataFrame(data_a)
        df_b = pd.DataFrame(data_b)
        
        self.df_raw = pd.concat([df_a, df_b], ignore_index=True).sort_values(by='Date').reset_index(drop=True)
        # MeetName is required by the evaluator to compose observation_id.
        self.df_raw['MeetName'] = 'Meet ' + self.df_raw['Date'].dt.strftime('%Y-%m')
        # Add dummy attempt columns so feature engineering doesn't fail
        for col in ['Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg']:
            self.df_raw[col] = 100
            
        self.df_raw['Athlete_ID'] = build_athlete_id(self.df_raw)
        
        self.feature_cols = [
            "Prev_Total",
            "Prev_Prev_Total",
            "Rolling_Mean_3",
            "Rolling_Std_3"
        ]

    def test_min_train_periods(self):
        # Test that evaluation doesn't run if there are not enough periods
        evaluator = WalkForwardEvaluator(models={}, baselines={}, min_train_periods=20)
        evaluator.evaluate(self.df_raw, feature_engineering, create_walk_forward_target, self.feature_cols)
        self.assertEqual(len(evaluator.metrics_results), 0)

        # Test that evaluation runs with sufficient periods
        evaluator = WalkForwardEvaluator(models={'Persistence': PersistenceBaseline()}, baselines={}, min_train_periods=5)
        evaluator.evaluate(self.df_raw, feature_engineering, create_walk_forward_target, self.feature_cols)
        
        self.assertGreater(len(evaluator.metrics_results), 0)

    def test_no_future_leakage_in_folds(self):
        # Let's manually check one fold
        train_df = self.df_raw[self.df_raw['Date'].dt.year < 2015]
        test_df_context = self.df_raw[self.df_raw['Date'].dt.year <= 2015]

        train_engineered = feature_engineering(train_df, athlete_col="Athlete_ID")
        test_engineered = feature_engineering(test_df_context, athlete_col="Athlete_ID")
        
        # The 'Prev_Total' for the first event of 2015 should be the last event of 2014
        athlete_a_test = test_engineered[(test_engineered['Date'].dt.year == 2015) & (test_engineered['Name'] == 'Athlete A')].iloc[0]
        prev_total_2015 = athlete_a_test['Prev_Total']
        
        athlete_a_2014_last = self.df_raw[(self.df_raw['Date'].dt.year == 2014) & (self.df_raw['Name'] == 'Athlete A')].iloc[-1]
        total_2014 = athlete_a_2014_last['TotalKg']
        
        self.assertEqual(prev_total_2015, total_2014)

if __name__ == '__main__':
    unittest.main()

    def test_missing_required_columns_raise_clear_error(self):
        """A missing structural column must be reported by name, not as a KeyError."""
        evaluator = WalkForwardEvaluator(models={}, baselines={}, min_train_periods=5)
        df = self.df_raw.drop(columns=['MeetName'])
        with self.assertRaises(ValueError) as ctx:
            evaluator.evaluate(df, feature_engineering, create_walk_forward_target, self.feature_cols)
        self.assertIn('MeetName', str(ctx.exception))

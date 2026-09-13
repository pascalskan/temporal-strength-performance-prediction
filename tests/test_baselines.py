import unittest
import pandas as pd
from src.models.baselines import PersistenceBaseline, RollingMeanBaseline, DriftBaseline

class TestBaselines(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'Prev_Total': [100, 110, 120, 130],
            'Prev_Prev_Total': [None, 100, 110, 120],
            'Rolling_Mean_3': [None, None, 110, 120]
        })

    def test_persistence_baseline(self):
        baseline = PersistenceBaseline()
        predictions = baseline.predict(self.data)
        expected = pd.Series([100, 110, 120, 130], name='Prev_Total')
        pd.testing.assert_series_equal(predictions, expected, check_names=False)

    def test_persistence_baseline_missing_feature(self):
        baseline = PersistenceBaseline()
        with self.assertRaises(ValueError):
            baseline.predict(pd.DataFrame({'Other_Col': [1, 2, 3]}))

    def test_rolling_mean_baseline(self):
        baseline = RollingMeanBaseline()
        predictions = baseline.predict(self.data)
        expected = pd.Series([None, None, 110, 120], name='Rolling_Mean_3')
        pd.testing.assert_series_equal(predictions, expected, check_names=False)

    def test_rolling_mean_baseline_missing_feature(self):
        baseline = RollingMeanBaseline()
        with self.assertRaises(ValueError):
            baseline.predict(pd.DataFrame({'Other_Col': [1, 2, 3]}))

    def test_drift_baseline(self):
        baseline = DriftBaseline()
        predictions = baseline.predict(self.data)
        expected = pd.Series([100.0, 120.0, 130.0, 140.0], name='Prev_Total')
        pd.testing.assert_series_equal(predictions, expected, check_names=False)

    def test_drift_baseline_missing_features(self):
        baseline = DriftBaseline()
        with self.assertRaises(ValueError):
            baseline.predict(pd.DataFrame({'Prev_Total': [1, 2, 3]}))

if __name__ == '__main__':
    unittest.main()
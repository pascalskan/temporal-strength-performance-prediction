import unittest
import pandas as pd
from src.data.identity import build_athlete_id


class TestAthleteIdentity(unittest.TestCase):

    def test_prefers_deterministic_athlete_id(self):
        df = pd.DataFrame({'AthleteID': [123], 'Name': ['A'], 'Sex': ['M']})
        self.assertEqual(build_athlete_id(df).iloc[0], '123')

    def test_falls_back_to_name_and_sex(self):
        df = pd.DataFrame({'Name': ['A'], 'Sex': ['M']})
        self.assertEqual(build_athlete_id(df).iloc[0], 'A_M')

    def test_unstable_attributes_are_excluded_from_the_key(self):
        """
        Regression guard. Weight class, age class and country change over an
        athlete's career; including them fragments single careers into multiple
        identities and discards ~41% of usable records, biased toward the long
        progressing careers that carry the temporal signal. OpenPowerlifting
        already disambiguates same-name athletes with a '#N' suffix, so these
        attributes add no separation of distinct people.
        """
        df = pd.DataFrame({
            'Name': ['A'], 'Sex': ['M'],
            'WeightClassKg': [85], 'AgeClass': ['20-23'], 'Country': ['UK'],
        })
        self.assertEqual(build_athlete_id(df).iloc[0], 'A_M')

    def test_career_spanning_weight_and_age_classes_stays_one_athlete(self):
        df = pd.DataFrame({
            'Name': ['A', 'A', 'A'],
            'Sex': ['M', 'M', 'M'],
            'WeightClassKg': [83, 83, 93],
            'AgeClass': ['20-23', '24-34', '24-34'],
        })
        self.assertEqual(build_athlete_id(df).nunique(), 1)

    def test_disambiguated_names_stay_separate_athletes(self):
        df = pd.DataFrame({'Name': ['Tony Nguyen #1', 'Tony Nguyen #12'], 'Sex': ['M', 'M']})
        self.assertEqual(build_athlete_id(df).nunique(), 2)

    def test_fails_without_a_usable_strategy(self):
        for bad in ({'Name': ['A']}, {'Sex': ['M']}, {'SomeOtherCol': [1]}):
            with self.assertRaises(ValueError):
                build_athlete_id(pd.DataFrame(bad))

    def test_explicit_override_is_honoured(self):
        df = pd.DataFrame({'Name': ['A'], 'Sex': ['M'], 'Federation': ['IPF']})
        self.assertEqual(build_athlete_id(df, ['Name', 'Federation']).iloc[0], 'A_IPF')

    def test_explicit_override_rejects_missing_columns(self):
        df = pd.DataFrame({'Name': ['A'], 'Sex': ['M']})
        with self.assertRaises(ValueError):
            build_athlete_id(df, ['Name', 'NotAColumn'])

    def test_nans_become_unknown(self):
        df = pd.DataFrame({'Name': ['A', 'B'], 'Sex': ['M', None]})
        ids = build_athlete_id(df)
        self.assertEqual(ids.iloc[0], 'A_M')
        self.assertEqual(ids.iloc[1], 'B_Unknown')


if __name__ == '__main__':
    unittest.main()

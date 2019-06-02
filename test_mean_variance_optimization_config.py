from stratdev.signals.configs.mean_variance_optimization_config import MeanVarianceOptimizationConfig
from stratdev.cdefs.defines import Format_t

from unittest.mock import patch
import numpy as np


class MockCommonSignalConfig:
    def __init__(self, is_valid_secid, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.json_string = kwargs

        self._valid_secid = is_valid_secid

    def is_valid_secid(self, secid):
        return self._valid_secid


class MockSecurityNameIndexer:
    @staticmethod
    def GetUniqueInstance():
        return MockSecurityNameIndexer()


def get_config():
    return {
        'allocation_deviation_penalty': 0.0,
        'benchmark_allocation_target_risk': 10.0,
        'benchmark_allocation_update_frequency': 5,
        'benchmark_market_cap_deviation_penalty': 0.01,
        'correlation_lookback_period': 252,
        'expected_return_indicator': [['default', 'DailyTrend', '1250']],
        'ind_transform': 'negative',
        'maximum_allocation_factor': 8.0,
        'maximum_leverage': 1.0,
        'name': 'MeanVarianceOptimization',
        'rebalancing': ['SimpleAverageRebalancingManager', 5],
        'regularization_threshold': 0.0,
        'risk_model': ['name', 'AverageStdevRiskModel', 'periods', '21', '63', '252', 'average_type', 'exponential'],
        'securities': ['VTI', 'BND'],
        'target_risk': 10.0,
        'target_risk_multiplier': [0.6, 1.4],
        'max_turnover': 1000.0,
    }


def get_default_aswellas_all_config():
    return {
        'target_risk':
        10.0,
        'maximum_leverage':
        1.0,
        'risk_model': ['name', 'AverageStdevRiskModel', 'periods', '42', '126', '378', 'average_type', 'exponential'],
        'rebalancing': ['SimpleAverageRebalancingManager', 5],
        'allocation_deviation_penalty':
        1.0,
        'economic_data': [
            'NonFarmPayroll', 'GrossDomesticProduct', 'Inflation', 'ShillerPERatio',
            'MerrillLynchCorporateBondsRiskSpread', 'BookValuePerShare', 'LogDividendPriceRatio', 'TreasuryBillRate',
            'LongTermYield', 'TermSpread', 'VIX', 'HousingPriceIndex'
        ],
        'expected_return_indicator':
        [['default', 'NearestNeighbour', '500', '0.005', '1000', 'PrincipalComponent', '6', '63', '500', 'all']],
        'name':
        'MeanVarianceOptimization',
        'dynamic_benchmark': {
            'mandate': 'Flagship',
            'bond_yield_weight': 1,
            'covariance': {
                'correlation': {
                    'lookback': 252
                },
                'risk_model': {
                    'name': 'AverageStdevRiskModel',
                    'periods': [63, 252, 1008],
                    'average_type': 'exponential'
                }
            },
            'allocation_update_threshold': 0.05,
            'yields_weights_regularization_threshold': 1
        },
        'maximum_allocation_factor':
        3.0,
        'target_risk_multiplier': [0.6, 1.4],
        'correlation_lookback_period':
        252.0,
        'regularization_threshold':
        0.0,
        'securities': ['VTI', 'BND', 'VXUS', 'VWO', 'VWOB', 'BNDX', 'VNQ', 'VNQI'],
    }


class TestMeanVarianceOptimizationConfig:
    @patch('signals.configs.mean_variance_optimization_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    @patch('signals.configs.validate_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    def test_load_json_config_should_load_correct_values(self, mock_sni1, mock_sni2):
        cnf = get_config()
        cnf['config_format'] = Format_t.Json
        cnf['expected_return_indicator'] = [['default', 'DailyTrend', '1250']]
        config = MockCommonSignalConfig(True, **cnf)
        mvo_config = MeanVarianceOptimizationConfig(config)

        # The default expected_return indicator/params to be used
        assert mvo_config.default_expected_return_indicator_info is not None

        # Ordering of 0th and 1st element has been reversed.
        assert all(
            np.array(mvo_config.default_expected_return_indicator_info.params) == np.array(
                ['DailyTrend', 'default', '1250']))

        # The lookback for computing the correlation matrix
        assert mvo_config.correlation_lookback_period == cnf['correlation_lookback_period']
        assert mvo_config.regularization_threshold == cnf['regularization_threshold']

        # for penalizing deviation from benchmark allocation
        assert mvo_config.allocation_deviation_penalty == cnf['allocation_deviation_penalty']
        assert mvo_config.num_samples == cnf.get('num_samples', 1)
        # No allcation constraint.
        assert len(mvo_config.allocation_constraints) == 0
        # No economic indicator
        assert len(mvo_config.economic_indicator_list) == 0
        # No beta constraint
        assert not ('list' in mvo_config.beta_constraints)
        # Security wise expected_return indicator info
        assert len(mvo_config.security_expected_return_indicator_info_list) == 0

    @patch('signals.configs.mean_variance_optimization_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    @patch('signals.configs.validate_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    def test_load_expected_returns_indicator_default_type(self, mock_sni1, mock_sni2):
        """
        Test the case when expected_return_indicator is `default`.
        """
        cnf = get_config()
        cnf['config_format'] = Format_t.Json
        cnf['expected_return_indicator'] = [['default', 'DailyTrend', '1250']]
        config = MockCommonSignalConfig(True, **cnf)
        assert MeanVarianceOptimizationConfig(config).default_expected_return_indicator_info is not None

    @patch('signals.configs.mean_variance_optimization_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    @patch('signals.configs.validate_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    def test_load_expected_returns_indicator_all_type(self, mock_sni1, mock_sni2):
        """
        When expected_return_indicator has 'all' as last token, it should have economic data.
        This test tests for the case when economic data is present and when it is absent.
        """
        cnf = get_config()
        cnf['config_format'] = Format_t.Json
        cnf['expected_return_indicator'] = [[
            'default', 'NearestNeighbour', '500', '0.005', '1000', 'PrincipalComponent', '6', '63', '500', 'all'
        ]]
        config = MockCommonSignalConfig(True, **cnf)
        try:
            MeanVarianceOptimizationConfig(config)
            raise Exception('Economic data was not present. Still got instantiated.')
        except BaseException:
            pass

        # After giving economic data, it should pass.
        cnf['economic_data'] = ['NonFarmPayroll']
        config = MockCommonSignalConfig(True, **cnf)
        MeanVarianceOptimizationConfig(config)

    @patch('signals.configs.mean_variance_optimization_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    @patch('signals.configs.validate_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    def test_load_expected_returns_indicator_default_aswellas_all_type(self, mock_sni1, mock_sni2):
        """
        Test the case when expected_return_indicator is both `default` and `all`.
        """
        cnf = get_default_aswellas_all_config()
        cnf['config_format'] = Format_t.Json
        config = MockCommonSignalConfig(True, **cnf)
        assert MeanVarianceOptimizationConfig(config).default_expected_return_indicator_info is not None

    @patch('signals.configs.mean_variance_optimization_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    @patch('signals.configs.validate_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    def test_load_expected_returns_indicator_security_specific_type(self, mock_sni1, mock_sni2):
        """
        This test tests load_expected_returns_indicator() when expected_return_indicator is specific to a security
        It also has multiple expected_return_indicator
        """
        cnf = get_config()
        cnf['config_format'] = Format_t.Json
        expected_return_indicators = [
            ['default', 'DailyTrend', '1250'],
            [
                'BIV', 'ExpectedLongTermBondYield', 'Ridge', '126', '126', '882', '0.0001', '252', 'USA5Y', 'USA7Y',
                'USA10Y'
            ],
            [
                'BLV', 'ExpectedLongTermBondYield', 'Ridge', '126', '126', '882', '0.0001', '252', 'USA10Y', 'USA20Y',
                'USA30Y'
            ],
        ]

        cnf['expected_return_indicator'] = expected_return_indicators
        config = MockCommonSignalConfig(True, **cnf)
        MeanVarianceOptimizationConfig(config)

    @patch('signals.configs.mean_variance_optimization_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    @patch('signals.configs.validate_config.SecurityNameIndexer', side_effect=MockSecurityNameIndexer)
    def test_load_beta_constraints(self, mock_sni1, mock_sni2):
        cnf = get_config()
        cnf['config_format'] = Format_t.Json
        cnf['beta_constraints'] = ['some_method', 'VT', '0', '0', '1', 'BND', '0', '0', '1']
        config = MockCommonSignalConfig(True, **cnf)
        MeanVarianceOptimizationConfig(config)

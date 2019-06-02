from datetime import date

import numpy as np
import pandas as pd

from qdata.core.cdefs.security_definitions import SecurityDefinitions
from stratdev.fasttest.position_sizing.base_position_sizing import BasePositionSizing
from stratdev_utils.asset_allocation_modules.risk_budgeting_allocator import RiskBudgetingAllocator
from stratdev_utils.asset_allocation_modules.risk_budgeting_allocator_config import RBAConfig
from stratdev_utils.asset_allocation_modules.risk_budgeting_allocator_log_max import RiskBudgetingAllocatorLogMax


class RiskBudgetingPositionSizing(BasePositionSizing):
    def __init__(self, config: dict, unnormalized_indicator_df: pd.DataFrame):
        """
        Args:
            config: signal config
            unnormalized_indicator_df: unnormalized indicator data. Note that is it normalized in the base class.
        """
        BasePositionSizing.__init__(self, config, unnormalized_indicator_df, None)

        self.validate_config(self.config)

        self.risk_factors = None
        # Instantiate RBA Allocator
        inputs = {}
        inputs['risk_config'] = config['risk_config']
        inputs['previous_weights'] = None
        inputs['max_leverage'] = config['maximum_leverage']
        inputs['min_allocation'] = config['minimum_allocation']
        inputs['max_allocation'] = config['maximum_allocation']
        inputs['target_risk'] = config['target_risk']
        inputs['display_opt_output'] = config.get('display_opt_output', False)
        inputs['initialize_with_previous_weights'] = config.get('initialize_with_previous_weights', False)
        inputs['risk_factors'] = self.compute_risk_factors(self.indicator_df, config['marketcap_method'])
        inputs['use_log_max_optimization'] = config.get('use_log_max_optimization', 0)
        self._rba_config = RBAConfig(inputs)
        self._rba_obj = None

    def validate_config(self, config: dict):
        '''
        Checks that the required parameters should be present in the config
        '''
        assert 'marketcap_method' in config, "marketcap_method missing"
        assert 'name' in config['marketcap_method'], "name attribute missing in marketcap_method"
        assert config['marketcap_method']['name'] in ['constant'], 'invalid value of name atribute of marketcap_method'

        if config['marketcap_method']['name'] == 'constant':
            assert abs(1.0 - sum(config['marketcap_method']['sector_weight_map'].values())) < 0.01
        assert 'risk_config' in config, "risk config missing"
        assert 'maximum_leverage' in config, "max_leverage missing"
        assert 'minimum_allocation' in config, "min_allocation missing"
        assert 'maximum_allocation' in config, "max_allocation missing"
        assert 'target_risk' in config, "target_risk missing"

    def compute_risk_factors(self, indicator_df: pd.DataFrame, marketcap_method: dict) -> pd.DataFrame:
        '''
        Computes risk factor from indicator df and market cap info
        Example:
        indicator_df
                                     ES_1  ZN_1
                         date
                         2018-01-01   1.0   0.5
                         2018-01-02   1.5  -0.5

        market_cap_df
                        weight
                        shortcode
                        ES_1          0.6
                        ZN_1          0.4

        risk_factor_df
                                    ES_1  ZN_1
                        date
                        2018-01-01   0.6   0.2
                        2018-01-02   0.9  -0.2

        '''

        # Parse market cap fraction per sector to get market cap fraction per product
        # assuming equal allocation in each sector
        if self.config['marketcap_method']['name'] == 'constant':
            # If no indicator is specified, create a dummy indicator dataframe of all ones.
            if indicator_df.empty:
                index = pd.date_range(self._start_date, self._end_date).date
                col = self.config['product_list']
                data = np.ones((len(index), len(col)))
                indicator_df = pd.DataFrame(data, columns=col, index=index)

            products = indicator_df.columns.tolist()
            sector_weight_map = self.config['marketcap_method']['sector_weight_map']

            # Multi index df with ('date', 'product') being the two levels.
            valid_days_filter = self._trading_days_filter.stack()

            # NOTE: The expression `valid_days_filter is True` is incorrect.
            # Select only those days when the product was traded
            valid_days_filter = valid_days_filter[valid_days_filter == True]  # noqa :E712

            ones_df = pd.DataFrame([], index=valid_days_filter.index)
            ones_df.reset_index('product', inplace=True)
            products_sector_map = {s: SecurityDefinitions.GetContractSectorStr(s) for s in products}
            ones_df['sector'] = ones_df['product'].map(products_sector_map)
            ones_df['weight'] = ones_df['sector'].map(lambda sector: sector_weight_map[sector])

            ones_df.index.name = 'date'
            ones_df = ones_df.reset_index()
            ones_df['date_copy'] = ones_df['date']
            ones_df = ones_df.set_index(['date', 'product'])[['sector', 'weight', 'date_copy']]
            # for each day, for each sector, equally divide the weights among products so that the sum of weights of
            # products for a particular sector is equal to sector_weight_map[sector] for each day.
            marketcap_df = ones_df.groupby(['date_copy', 'sector']).transform(lambda x: x / len(x))

            # products are made the columns
            marketcap_df = marketcap_df['weight'].unstack()

        # Multiply indicator df with market cap fractions to get the risk budget per product
        self.risk_factors = indicator_df * marketcap_df
        try:
            self.risk_factors.index = self.risk_factors.index.date
        except Exception:
            pass
        return self.risk_factors

    def _compute(self) -> pd.DataFrame:
        """
        Returns:
        Pandas DF with index as timestamp -> columns as products, and values as cell values
        """
        if self._rba_config.use_log_max_optimization:
            self._rba_obj = RiskBudgetingAllocatorLogMax(self._rba_config)
        else:
            self._rba_obj = RiskBudgetingAllocator(self._rba_config)

        self.allocation_df = self._rba_obj.get_allocation_df(logging=self.config.get('logging', False))
        # For, initial set of days for which there is no signal, drop those days
        self.allocation_df = self.allocation_df.fillna(method='ffill').dropna(how='all', axis=0)

        self._perform_common_operations_end()
        return self.allocation_df


if __name__ == '__main__':
    risk_config = {
        'correlation': {
            'lookback': 512,
        },
        'risk_model': {
            'periods': [63, 252, 1024],
            'name': 'AverageStdevRiskModel',
            'average_type': 'exponential',
        },
    }

    config = {
        'product_list': ['ES_1', 'ZN_1'],
        'use_log_max_optimization': 1,
        'risk_config': risk_config,
        'maximum_leverage': 4.0,
        'maximum_allocation': 4.0,
        'minimum_allocation': -4.0,
        'target_risk': 10.0,
        'marketcap_method': {
            'sector_weight_map': {
                'Equity': 0.6,
                'FixedIncome': 0.4
            },
            'name': 'constant'
        },
        'start_date': date(2018, 1, 1),
        'end_date': date(2018, 1, 2)
    }

    indicator_df = pd.DataFrame(
        [(date(2018, 1, 1), 1.0, 0.5), (date(2018, 1, 2), 1.5, -0.5)], columns=['date', 'ES_1',
                                                                                'ZN_1']).set_index('date')

    risk_budgeting_position_sizing = RiskBudgetingPositionSizing(config, indicator_df)
    allocation_df = risk_budgeting_position_sizing.allocate()
    print(allocation_df)

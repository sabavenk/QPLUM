'''Testing module for MaxSharpeAllocator'''
from datetime import date
import pytest
import pandas
import numpy as np

# Qplum Imports
from qdata import DailyData
from stratdev_utils.asset_allocation_modules.risk_budgeting_allocator_config import RBAConfig
from stratdev_utils.asset_allocation_modules.max_sharpe_allocator import MaxSharpeAllocator


@pytest.mark.docker
def test_positive_risk_budgets():
    symbol_list = ['ES_1', 'ZN_1', 'GC_1', 'HG_1', 'CL_1', 'ZM_1']
    risk_allocations = [0.5, 0.2, 0.075, 0.075, 0.075, 0.075]

    # Dates for contructing Portfolio

    earliest_data_availability_date = date(1998, 1, 1)
    start_date = earliest_data_availability_date
    end_date = date(2018, 11, 1)

    # Making sure sum of risk allocations is 1
    var_allocations = np.array(risk_allocations)
    var_allocations = var_allocations / np.sum(np.abs(var_allocations))

    # Creating a dataframe with each risk_allocations row for each date for which we have prices data
    prices_df = DailyData(symbol_list, start_date, end_date).raw_data().xs('close', level=1, axis=1)
    var_allocations_df = pandas.DataFrame(
        [var_allocations] * len(prices_df), columns=symbol_list, index=prices_df.index)

    # Other constraints on new Portfolio
    annual_nominal_target_risk = 10
    max_leverage = 4.0
    min_allocation = 0
    max_allocation = 2

    # Qplum internal risk_model_parameter
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

    ms_inputs = {}
    ms_inputs['risk_config'] = risk_config
    ms_inputs['risk_factors'] = var_allocations_df
    ms_inputs['max_leverage'] = max_leverage
    ms_inputs['min_allocation'] = min_allocation
    ms_inputs['max_allocation'] = max_allocation
    ms_inputs['target_risk'] = annual_nominal_target_risk
    ms_inputs['initial_weights'] = None
    ms_inputs['to_allow_changing_target_risk'] = False
    ms_inputs['to_match_allocation_and_risk_factor_signs'] = False

    ms_config = RBAConfig(ms_inputs)
    ms_obj = MaxSharpeAllocator(ms_config)
    del (ms_config)

    # Testing allocation equivalence on a date
    test_date = date(2018, 10, 1)
    saved_alloc_on_date = np.array([
        0.8587056210359881, 1.0796267708955245, -0.30007494197450485, 0.13154399751195706, -0.014645197813824713,
        -0.01609338371357999
    ])
    new_alloc_on_date = ms_obj.get_allocation_on_date(test_date)

    if not np.allclose(new_alloc_on_date, saved_alloc_on_date):
        print('Expected allocation = {}'.format(saved_alloc_on_date))
        print('Received allocation = {}'.format(new_alloc_on_date))
        raise ValueError('Allocation of the MSAllocator does not match what we were expecting.')

    # Once I get a sum of allocation from @sabavenk, I'll add this part
    allocation_df = (ms_obj.get_allocation_df(logging=False))
    new_alloc_sum = allocation_df.sum(axis=0).values
    saved_alloc_sum = np.array([3166.77326666, 3497.36542676, -48.86752613, 7.19924706, 38.33135736, 203.52976143])
    if not ((len(new_alloc_sum) == len(saved_alloc_sum)) and np.allclose(saved_alloc_sum, new_alloc_sum)):
        print('Expected vsum of allocation = {}'.format(saved_alloc_sum))
        print('Received vsum of allocation = {}'.format(new_alloc_sum))
        print(ms_inputs)
        print('Dates: {} to {}'.format(allocation_df.index[0], allocation_df.index[-1]))
        raise ValueError('Vertical sum of allocations of the MSAllocator does not match what we were expecting.')


def __main__():
    test_positive_risk_budgets()


__main__()

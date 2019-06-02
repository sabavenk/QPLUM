'''Testing module for MeanVarCVXAllocator'''

from datetime import date

import numpy as np
import pandas as pd
import pytest
# Qplum Imports
from stratdev_utils.asset_allocation_modules.mvo_cvx_allocator import MeanVarCVXAllocator
from stratdev_utils.asset_allocation_modules.mvo_cvx_config import MeanVarCVXConfig


@pytest.mark.docker
def test_mvo_cvx():

    # Defining the portfolio universe
    product_list = [
        'AAXJ', 'AGG', 'BIV', 'BND', 'BNDX', 'BSV', 'CWB', 'DBC', 'EMB', 'EWI', 'EWJ', 'EWL', 'EWU', 'HYG', 'IEF',
        'JNK', 'LQD', 'QQQ', 'SHV', 'STIP', 'TIP', 'TLT', 'VCIT', 'VCSH', 'VGIT', 'VGK', 'VTI', 'VTIP', 'VWO', 'VWOB'
    ]

    # dates
    start_date = date(1995, 1, 2)
    end_date = date(2018, 11, 5)

    # Setting a date on which we will test the allocation
    test_date = date(2018, 11, 19)

    # dataframe of exp_ret values on test_date
    # Copied from notebook:
    #    ProdNotebooks/CVX_Optimization_Research/Long_Only_MVO/MVO_CVX_Allocator.ipynb#Notes-on-MVO-Logic
    expret_vals = [[
        0.0, 4.2205e-05, 3.923110000000001e-05, 5.12231e-05, -0.0004129902, 9.01807e-05, 0.0, 0.0005213682,
        -0.0001524339, -0.00016177299999999998, -0.00015676320000000002, 0.0002283432, 9.34302e-05, -1.21917e-05,
        0.00015408160000000002, 0.0, -4.9186499999999994e-05, -4.07612e-05, -1.40793e-05, 0.0, 0.0001463235,
        0.00015637770000000002, 0.0, 3.67473e-05, 6.314620000000001e-05, 0.0001660699, -0.0003663757,
        0.00017648069999999999, 0.0010981896, -0.00016003880000000002
    ]]

    expret_df = pd.DataFrame(expret_vals, columns=product_list, index=[test_date])

    # configuration for calculating covariance matrices
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

    # other input values
    threshold_risk = 0.1 / np.sqrt(252)  # Annualized 10% stdev
    target_risk = 0.14 / np.sqrt(252)  # Annualized 14% stdev

    # setup configuration
    config_inputs = {}
    config_inputs['risk_config'] = risk_config
    config_inputs['expected_return_indicator_df'] = expret_df
    config_inputs['threshold_risk'] = threshold_risk
    config_inputs['target_risk'] = target_risk
    config_inputs['start_date'] = start_date
    config_inputs['end_date'] = end_date
    config_inputs['product_list'] = product_list

    mvo_cvx_config = MeanVarCVXConfig(config_inputs)
    mvo_cvx_obj = MeanVarCVXAllocator(mvo_cvx_config)

    # test on test_date
    actual_alloc = [
        2.995728519200449e-09, 3.915861865524969e-10, 7.727488372388348e-10, 9.092879480218302e-10,
        3.2179383351557617e-09, 0.002078791792248206, 2.995728519200425e-09, 4.635132501380115e-10,
        1.7094162356237784e-10, 1.1950690662431534e-10, 3.127613184501551e-10, 0.00726926849162644,
        1.0473431971435435e-09, 0.955920956419266, 1.5265374900089823e-09, 2.9957285192004393e-09, 7.97369543120461e-10,
        1.2961271434311357e-10, 4.703312932449714e-09, 2.9957285192004344e-09, 8.549804346313119e-10,
        6.274435235306893e-10, 2.9957285192004584e-09, 5.673757854893268e-09, 2.236419932370802e-09,
        0.03434883610297996, 1.2411883432353966e-09, 1.2969748654172543e-09, 0.00038210349456479353,
        2.2274020258383316e-09
    ]

    test_alloc = list(mvo_cvx_obj.get_allocation_on_date(test_date))

    if not np.allclose(actual_alloc, test_alloc, equal_nan=True):
        print('Expected allocation = {}'.format(actual_alloc))
        print('Received allocation = {}'.format(test_alloc))
        raise ValueError('Allocation of the MeanVarCVXAllocator does not match what we were expecting.')


if __name__ == "__main__":
    test_mvo_cvx()

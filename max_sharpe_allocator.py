'''
This module uses an approach to allocation that is inspired by Grinold and Kahn section 3.2
We are finding constant weights that would maximize the Sharpe Ratio.

'''
from copy import deepcopy
import datetime
from typing import List

import numpy as np
from numpy.linalg import inv
import pandas as pd

# Qplum Imports
from stratdev_utils.asset_allocation_modules.risk_budgeting_allocator_config import RBAConfig
from stratdev_utils.asset_allocation_modules.matrix_utils import get_correlation_matrix_from_covariance
from stratdev_utils.performance.performance_defines import NUM_TRADING_DAYS_IN_YEAR
from stratdev_utils.risk_models.covariance import Covariance


def find_constant_addition(w1: np.array, worig: np.array):
    '''Find a constant multiplier k such that (w1 + k * worig) have the same signs as worig'''
    k = np.max(-np.multiply(w1, worig) / np.multiply(worig, worig))
    return k


class MaxSharpeAllocator():
    def __init__(self, ms_config: RBAConfig, verbose=False) -> None:
        self.allocator_config = ms_config
        if ms_config.is_valid_rba_config():
            self.cov_obj = Covariance(ms_config.sec_list, ms_config.risk_config_dict)
        self.verbose = verbose
        self.start_date = self.allocator_config.risk_factor_df.index[0]
        self.end_date = self.allocator_config.risk_factor_df.index[-1]
        # we are working with annualized log covariance_matrix
        self.covariance_panel = self.cov_obj.get_covariance_panel_to_use(
            self.allocator_config.sec_list, self.start_date, self.end_date, is_nominal=False)
        self.covariance_panel = self.covariance_panel * NUM_TRADING_DAYS_IN_YEAR
        # relying on input riskfactor for dates instead of covariance panel !
        self.allocation_df = pd.DataFrame(index=self.covariance_panel.items, columns=self.allocator_config.sec_list)
        self.previous_weights = deepcopy(self.allocator_config.previous_weights)

    def get_allocation_df(self, logging: bool = False) -> pd.DataFrame:
        for t_date in self.allocation_df.index:
            if t_date in self.allocator_config.risk_factor_df.index:
                self.allocation_df.loc[t_date, self.allocator_config.sec_list] = self.get_allocation_on_date(
                    t_date, logging=logging)
        return self.allocation_df

    def get_allocation_on_date(self, input_date: datetime.date, logging: bool = False) -> List[float]:
        """
        we are doing per date for now, the code will be easier to read
        we are trying to minimize the difference between
        risk_contribution defined as: Wn x 1 * (Covn x n * Wn x 1)
        risk_budget defined as: risk_factors * target_risk
        specifically we minimize the sum_of_errors (one error per one sec)
        """
        # internal variables
        covariance_matrix = self.covariance_panel.loc[input_date, self.allocator_config.sec_list].values
        risk_budget_fraction = self.allocator_config.risk_factor_df.loc[input_date, self.allocator_config.
                                                                        sec_list].values

        # Checking that the arguments have the right dimensions.
        assert len(covariance_matrix.shape) == 2, 'Covariance matrix must be 2-D'
        assert (risk_budget_fraction.shape[0] == covariance_matrix.shape[0]
                ), 'Shapes of risk_budget_fraction and covariance_matrix should match'

        # Filling nans with 0
        covariance_matrix[np.isnan(covariance_matrix)] = 0.0
        risk_budget_fraction[np.isnan(risk_budget_fraction)] = 0.0

        corr_matrix, stdev_array = get_correlation_matrix_from_covariance(covariance_matrix)
        inv_stdev = 1 / stdev_array
        del (stdev_array)

        risk_budget_fraction_abs = np.abs(risk_budget_fraction)

        # Allow scaling target risk based on magnitude of risk budget
        target_risk = self.allocator_config.target_risk * np.sum(
            risk_budget_fraction_abs
        ) if self.allocator_config.to_allow_changing_target_risk else self.allocator_config.target_risk

        # Normalise risk_budget_fraction to get expected Sharpe Ratios
        # Since scale does not matter, we are making them sum up to 1.
        prop_to_expected_sharpe = risk_budget_fraction / np.sum(risk_budget_fraction_abs)

        # weights = matrix-inverse(corr-matrix) * pairwise-multiplication ( prop_to_expected_sharpe, (1/stdev) )
        # Note that pairwise-multiplication ( prop_to_expected_sharpe, (1/stdev) ) are also the
        # unscaled_corr_agnostic_weights. If the correlation matrix inverse is np.diag(ones) then
        # unscaled_corr_agnostic_weights would be optimal weights
        unscaled_corr_agnostic_weights = np.multiply(prop_to_expected_sharpe, inv_stdev)
        unscaled_weights = np.squeeze(np.asarray(inv(corr_matrix) * np.asmatrix(unscaled_corr_agnostic_weights).T))

        # Compute how much this set of weights needs to be multiplied with
        # to achieve the target_risk
        portfolio_multiplier = target_risk / (np.sqrt(
            np.dot(unscaled_weights, np.dot(covariance_matrix, unscaled_weights))))
        # Scale weights
        scaled_weights = unscaled_weights * portfolio_multiplier
        opt_weights = scaled_weights

        # If sign of scaled_weights differs from sign of risk_budget_fraction
        if self.allocator_config.to_match_alloc_to_rb and (not (scaled_weights * risk_budget_fraction >= 0).all()):
            no_corr_port_mult = target_risk / (np.sqrt(
                np.dot(unscaled_corr_agnostic_weights, np.dot(covariance_matrix, unscaled_corr_agnostic_weights))))
            scaled_ca_wts = unscaled_corr_agnostic_weights * no_corr_port_mult
            assert (np.all(np.sign(scaled_ca_wts) * np.sign(risk_budget_fraction) >= 0)
                    ), 'Not sure how this failed. This should not be necessary as inv_stdev is positive only'
            # wf = w + k * w_ca such that wf sign matches with risk_budget_fraction
            k_out = find_constant_addition(scaled_weights, scaled_ca_wts)
            # make final unscalled weights
            unscaled_same_sign_weights = (scaled_weights + k_out * scaled_ca_wts)
            if not (np.all(np.sign(unscaled_same_sign_weights) * np.sign(risk_budget_fraction)) >= 0):
                raise ValueError('Now signs should match but they don\'t!')
            # Compute how much this set of weights needs to be multiplied with
            # to achieve the target_risk
            ss_port_mult = target_risk / (np.sqrt(
                np.dot(unscaled_same_sign_weights, np.dot(covariance_matrix, unscaled_same_sign_weights))))
            # Scale weights
            scaled_same_sign_weights = unscaled_same_sign_weights * ss_port_mult
            # Set these new weights as return value
            opt_weights = scaled_same_sign_weights

        if logging:
            # Check Deviation from target risk
            risk_deviation_from_target = (
                100.0 * (target_risk - (np.sqrt(np.dot(opt_weights, np.dot(covariance_matrix, opt_weights))))))
            if risk_deviation_from_target > 0.1:
                print('\nAnnualized Risk deviation from target(%) on {}: {}'.format(input_date,
                                                                                    risk_deviation_from_target))

            # Signs of weights and indicators do not match
            if not (opt_weights * risk_budget_fraction >= 0).all():
                print('Signs of weights and indicators do not match on {}. Wts: {} Ind: {} Match: {}'.format(
                    input_date, opt_weights.round(4), risk_budget_fraction.round(4),
                    (opt_weights * risk_budget_fraction >= 0)))

        # Wherever we encounter nans, set them to 0
        # also adjust previous weights if all weights are 0
        opt_weights[np.isnan(opt_weights)] = 0
        if np.sum(np.abs(opt_weights)) == 0:
            opt_weights = np.copy(self.previous_weights)
        else:
            self.previous_weights = opt_weights
        return opt_weights

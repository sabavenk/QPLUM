import math
from datetime import date

import numpy as np
import cvxpy as cp  # convex optimizer package

from stratdev.cdefs.security_name_indexer import SecurityNameIndexer
from stratdev.computation_classes.dynamic_benchmark import DynamicBenchmark
from stratdev.computation_classes.market_cap_weights import MarketCapWeightsInfo
from stratdev.computation_classes.target_risk_manager import TargetRiskManager
from stratdev.indicators.indicator_list import Run_Mode, instantiate_indicator
from stratdev.signals.base_signal import BaseSignal
from stratdev.signals.configs.mean_variance_optimization_config import \
    MeanVarianceOptimizationConfig
from stratdev.utils.save_state_utils import (get_save_state_info_from_secid_values,
                                             get_secids_values_from_save_state_info)
from stratdev.indicators.indicator_list import Indicator_t


class LowerAllocationConstraint(object):
    def __init__(self, shortcodes, min_allocation):
        self.min_allocation = min_allocation
        self.shortcodes = shortcodes
        self.flag_array = np.empty(0)


class UpperAllocationConstraint(object):
    def __init__(self, shortcodes, max_allocation):
        self.max_allocation = max_allocation
        self.shortcodes = shortcodes
        self.flag_array = np.empty(0)


class BetaConstraint(object):
    def __init__(self, beta_instance, min_beta, max_beta, shortcode, secid):
        self.beta_instance = beta_instance
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.shortcode = shortcode
        self.secid = secid


class MeanVarianceCvxOpt(BaseSignal):
    def __init__(self, mode, **kwargs):
        if mode == Run_Mode.SIM:
            '''
            watch Reference to the watch
            progstate Reference to the common program state class
            The unique signal id
            '''
            self.watch = kwargs['watch']
            self.commoncache = kwargs['commoncache']
            self.sgid = kwargs['sgid']
            self.initialize_mvo_sim(mode, **kwargs)

        elif mode == Run_Mode.TEST:
            self.initialize_mvo_test(mode, **kwargs)
        else:
            raise AssertionError

    def initialize_mvo_test(self, mode, **kwargs):
        BaseSignal.__init__(self, mode, **kwargs)
        self.config = self.common_config
        self.mode = mode
        self.weights_dict = {}
        self.last_weights = None

        self.weights_variable = cp.Variable(self.common_config.get_num_securities())

        self.num_days = 0
        self.tr_constant = None
        self.initialize_alloc_constraints()

        self.compute_allocation()

    def initialize_mvo_sim(self, mode, **kwargs):
        BaseSignal.__init__(self, mode, **kwargs)
        self.config = MeanVarianceOptimizationConfig(self.common_config)
        self.num_days = 0
        self.objective_called = False
        self.tr_constant = None
        self.last_weights = None
        self.expected_return_indicators = []
        self.risk_model_value = np.empty(0)
        self.expected_return_indicator_value = np.empty(0)
        self.past_weights_optim = np.empty(0)
        self.cov_mat = np.empty((0, 0))
        self.bounds = []

        self.weights_variable = cp.Variable(self.common_config.get_num_securities())

        # self.correlation_instance = Correlation.GetUniqueInstance(
        #     self.watch, self.commoncache, self.config.correlation_lookback_period, self.load_datetime)
        # we will store the benchmark allocation of this signal's shortcodes
        self.benchmark_allocation = {}
        self.benchmark_score = 0.0

        self.initialize_alloc_constraints()

        self.is_ready = False

    def initialize_alloc_constraints(self):
        if self.config.allocation_deviation_penalty != 0.0:
            self.dyn_benchmark = DynamicBenchmark.GetUniqueInstance(self.all_shortcodes,
                                                                    self.common_config.dyn_benchmark_config)

        self.constraints = []
        self.constraint_names = []

        # Allocation bounds:
        self.lower_allocations = []
        self.upper_allocations = []
        for ac in self.config.allocation_constraints:
            self.lower_allocations.append(LowerAllocationConstraint(ac.shortcodes, ac.min_allocation))
            self.upper_allocations.append(UpperAllocationConstraint(ac.shortcodes, ac.max_allocation))

    def initialize_other_constraints(self):

        # Allocation Constraints: Ensure the allocation for each security is between the min and max allocation

        for idx, ac in enumerate(self.lower_allocations):
            self.constraints.append(cp.sum(self.weights_variable * ac.flag_array) >= ac.min_allocation)
            self.constraint_names.append('lower_alloc {}'.format(idx))
        for idx, ac in enumerate(self.upper_allocations):
            self.constraints.append(cp.sum(self.weights_variable * ac.flag_array) <= ac.max_allocation)
            self.constraint_names.append('upper_alloc {}'.format(idx))

        # LONG-ONLY Constraint:

        self.constraints.append(self.weights_variable >= 0)
        self.constraint_names.append('Non-Negative Weights')

        # Leverage Constraint: Ensures that the allocation weights adds up to max_leverage (1 in this case).

        self.leverage_constraint = (cp.sum(self.weights_variable) == self.common_config.maximum_leverage)
        self.constraints.append(self.leverage_constraint)
        self.constraint_names.append('Leverage')

        # Upper-Target-Risk Constraint: Ensures the allocation risk doesn't exceed the upper target limit in config.

        self.target_upper_stdev_daily_log_returns = math.sqrt(
            TargetRiskManager.get_daily_logret_target_variance(self.common_config.upper_target_risk))

        self.upper_target_risk_const = (cp.quad_form(self.weights_variable, np.asmatrix(self.cov_mat)) <=
                                        self.target_upper_stdev_daily_log_returns)

        self.constraints.append(self.upper_target_risk_const)
        self.constraint_names.append('Upper_Target_Risk')

    def objective_fn(self):
        """
        Summary:
        This defines the objective function w.r.t which the portfolio allocation is computed.

        Detailed Description:
        Using the class variables to gain access to the required inputs, this objective function we will minimize
        is calculated using the following:
        : term_1 (float): Expected return value of portfolio given the weight allocation.
        : term_2 (float): Ridge regularization penalty term to induce preference for spread-out weight allocations.
        : term_3 (float): Turnover penalty term incentivizing minimal deviation from the previous weight allocation.
        : term_4 (float): Benchmark deviation penalty term to disincentivize active shares.
        : term_5 (float): Expected return value of benchmark portfolio.
        : return (float): -term_1 + term_2 + term_3 + term_4 + term_5
        """

        eq_weights = np.array(
            [(1.0 if self.expected_return_indicator_value[i] > 0.0 else
              (-1.0 if self.expected_return_indicator_value[i] < 0.0 else 0.0)) * 1.0 / self.weights_variable.shape[0]
             for i in range(self.weights_variable.shape[0])])

        self.reg_constant = abs(np.sum(
            eq_weights * self.expected_return_indicator_value)) * self.config.regularization_threshold / np.sum(
                eq_weights * eq_weights)
        self.benchmark_constant = abs(
            np.sum(eq_weights * self.expected_return_indicator_value)
        ) * self.config.allocation_deviation_penalty / self.get_deviation_from_benchmark_allocation(eq_weights)
        if self.tr_constant is None:
            self.tr_constant = 0.1 * abs(np.sum(eq_weights * self.expected_return_indicator_value)) / np.sum(
                (eq_weights - self.last_weights) * (eq_weights - self.last_weights))
        elif (self.num_days % 63 == 0) and (not self.objective_called):
            self.objective_called = True
            self.tr_constant = 0.5 * (self.tr_constant + self.tr_constant *
                                      (self.get_turnover(63) / self.common_config.max_turnover))
            self.tr_constant = min(max(0.05, self.tr_constant), 10.0)

        if math.isnan(self.reg_constant):
            self.reg_constant = 0.0
        if math.isnan(self.tr_constant):
            self.tr_constant = 0.1
        if math.isnan(self.benchmark_constant):
            self.benchmark_constant = 0.0

        self.term_1 = -cp.sum(self.weights_variable * self.expected_return_indicator_value)
        self.term_2 = self.reg_constant * cp.sum(self.weights_variable * self.weights_variable)
        self.term_3 = self.tr_constant * cp.sum((self.weights_variable - self.last_weights)**2)
        self.term_4 = self.benchmark_constant * self.get_deviation_from_benchmark_allocation(self.weights_variable)
        self.term_5 = cp.sum(self.weights_variable) * self.benchmark_score

        self.objective = cp.Minimize(self.term_1 + self.term_2 + self.term_3 + self.term_4 + self.term_5)

    def signal_name(self) -> str:
        return ('MeanVarianceCvxOpt')

    def set_expected_return_risk_value(self, day: date):
        self.expected_return_indicator_value = self.get_expected_return_value_to_use_on_date(day)
        self.risk_model_value = self.get_expected_risk_value_to_use_on_date(day)
        self.is_ready = True

    @staticmethod
    def GetUniqueInstance(watch, commoncache, sgid, load_datetime=None):
        if sgid not in MeanVarianceCvxOpt.signal_map:
            MeanVarianceCvxOpt.signal_map[sgid] = MeanVarianceCvxOpt(
                Run_Mode.SIM, watch=watch, commoncache=commoncache, sgid=sgid, load_datetime=load_datetime)
        else:
            MeanVarianceCvxOpt.signal_map[sgid].process_new_load_datetime(load_datetime)
        return MeanVarianceCvxOpt.signal_map[sgid]

    # @brief Reset load_datetime of the signal and call indicator instances to reset their load_datetimes
    def reset_load_datetime(self, load_datetime):
        super(MeanVarianceCvxOpt, self).reset_load_datetime(load_datetime)
        for indicator in self.expected_return_indicators:
            indicator.process_new_load_datetime(self.load_datetime)
        # self.correlation_instance.process_new_load_datetime(self.load_datetime)

        # @brief Save the variables specific to the signal
    def save_signal_state(self, current_time):
        state = super(MeanVarianceCvxOpt, self).save_signal_state(current_time)
        state['past_weights_optim'] = get_save_state_info_from_secid_values(self.sec_weights_info.secids,
                                                                            self.past_weights_optim)
        state['last_weights'] = get_save_state_info_from_secid_values(self.sec_weights_info.secids, self.last_weights)

        state['num_days'] = self.num_days
        state['tr_constant'] = self.tr_constant
        state['benchmark_allocation'] = self.benchmark_allocation
        return state

    # @brief Save the variables specific to the signal
    def load_signal_state(self, load_datetime, state):
        super(MeanVarianceCvxOpt, self).load_signal_state(load_datetime, state)
        self.past_weights_optim = get_secids_values_from_save_state_info(
            state['past_weights_optim'], self.sec_weights_info.secids, numpy_type=True)[1]
        self.last_weights = get_secids_values_from_save_state_info(
            state['last_weights'], self.sec_weights_info.secids, numpy_type=True)[1]
        self.num_days = state['num_days']
        self.tr_constant = state['tr_constant']
        self.benchmark_allocation = state['benchmark_allocation']
        self.is_live = True

    # @brief Called when a new securiity starts trading on the exchange

    def on_new_security_added(self, secid):
        super(MeanVarianceCvxOpt, self).on_new_security_added(secid)
        shortcode = SecurityNameIndexer.GetUniqueInstance().get_shortcode_from_id(secid)
        self.sec_weights_info.secids.append(secid)
        sec_name_indexer = SecurityNameIndexer.GetUniqueInstance()
        self.shortcodes = [sec_name_indexer.get_shortcode_from_id(secid) for secid in self.sec_weights_info.secids]
        self.sec_weights_info.weights = np.append(self.sec_weights_info.weights, 0)
        self.past_weights_optim = np.append(self.past_weights_optim, 0.001)
        self.risk_model_value = np.append(self.risk_model_value, 0.0)
        self.expected_return_indicator_value = np.append(self.expected_return_indicator_value, 0.0)

        for la in self.lower_allocations:
            if shortcode in la.shortcodes:
                la.flag_array = np.append(la.flag_array, 1.)
            else:
                la.flag_array = np.append(la.flag_array, 0.)

        for ua in self.upper_allocations:
            if shortcode in ua.shortcodes:
                ua.flag_array = np.append(ua.flag_array, 1.)
            else:
                ua.flag_array = np.append(ua.flag_array, 0.)

        # Set bound
        # Commenting max_allocation as unused
        # max_allocation = (self.common_config.maximum_allocation_factor) * (
        #     self.common_config.maximum_leverage / self.past_weights_optim.shape[0])
        self.bounds = []

        # Set expected_return Indicator
        security_specific_indicator_found = False
        security_expected_return_indicator_info_list = self.config.security_expected_return_indicator_info_list
        # Try to find explicit indicator specification
        for indicator_info in security_expected_return_indicator_info_list:
            if indicator_info.secid == secid:
                security_specific_indicator_found = True
                self.indicator_cache_map[shortcode] = {
                    'config': {
                        'name': Indicator_t.reverse[indicator_info.name],
                        'params': indicator_info.params,
                        'rm_config': self.common_config.risk_model_config
                    },
                    'shortcode': shortcode,
                    'tag': 'RAW'
                }
                self.indicator_cache_map[shortcode]['values'] = {}
                self.expected_return_indicators.append(
                    instantiate_indicator(
                        self.watch,
                        self.commoncache,
                        indicator_info.name,
                        indicator_info.params,
                        risk_model_config=self.common_config.risk_model_config,
                        load_datetime=self.load_datetime))
                break
        # If indicator not explicitly specified for this security, use default
        if not security_specific_indicator_found:
            sec_name_indexer = SecurityNameIndexer.GetUniqueInstance()
            default_expected_return_indicator_params = list(
                self.config.default_expected_return_indicator_info.params)  # Copy the indicator params
            default_expected_return_indicator_params[1] = sec_name_indexer.shortcode_list[
                secid]  # Overwrite "default" with the security shortcode
            self.indicator_cache_map[shortcode] = {
                'config': {
                    'name': Indicator_t.reverse[self.config.default_expected_return_indicator_info.name],
                    'params': default_expected_return_indicator_params,
                    'rm_config': self.common_config.risk_model_config
                },
                'shortcode': shortcode,
                'tag': 'RAW'
            }
            self.indicator_cache_map[shortcode]['values'] = {}
            self.expected_return_indicators.append(
                instantiate_indicator(
                    self.watch,
                    self.commoncache,
                    self.config.default_expected_return_indicator_info.name,
                    default_expected_return_indicator_params,
                    risk_model_config=self.common_config.risk_model_config,
                    load_datetime=self.load_datetime))
        # update benchmark only if allocation penalty is non zero
        if self.config.allocation_deviation_penalty != 0.0:
            self.mkt_cap_info = MarketCapWeightsInfo(self.shortcodes)
            self.update_benchmark_allocation()

    # @brief Called when an existing security stops trading on the exchange
    def on_security_removed(self, secid):
        super(MeanVarianceCvxOpt, self).on_security_removed(secid)
        try:
            idx = self.sec_weights_info.secids.index(secid)
            del self.sec_weights_info.secids[idx]
            del self.expected_return_indicators[idx]
            del self.bounds[idx]
            self.sec_weights_info.weights = np.delete(self.sec_weights_info.weights, idx)
            sec_name_indexer = SecurityNameIndexer.GetUniqueInstance()
            self.shortcodes = [sec_name_indexer.get_shortcode_from_id(secid) for secid in self.sec_weights_info.secids]
            self.allocation_signs = np.delete(self.allocation_signs, idx)
            self.past_weights_optim = np.delete(self.past_weights_optim, idx)
            self.risk_model_value = np.delete(self.risk_model_value, idx)
            self.expected_return_indicator_value = np.delete(self.expected_return_indicator_value, idx)

            for la in self.lower_allocations:
                la.flag_array = np.delete(la.flag_array, idx)

            for ua in self.upper_allocations:
                ua.flag_array = np.delete(ua.flag_array, idx)

            self.bounds = []
            # update benchmark only if allocation penalty is non zero
            if self.config.allocation_deviation_penalty != 0.0:
                self.mkt_cap_info = MarketCapWeightsInfo(self.shortcodes)
                self.update_benchmark_allocation()
        except ValueError:
            raise AssertionError("MeanVarianceOptimization Signal : on_security_removed, secid not found")

    def update_benchmark_allocation(self):
        """
        Function to update the benchmark allocation depending upon the target risk and covariance matrix
        """
        # get the benchmark allocation from market cap and the risk target expressed as percentage of market cap risk
        self.benchmark_allocation = self.dyn_benchmark.get_allocation_to_use_on_date(self.watch.current_date)

    def get_benchmark_allocation_score(self):
        """
        function that returns a weighted average of expected return scores of broader ETFs using benchmark allocation as
        weights
        """
        if self.config.allocation_deviation_penalty == 0.0:
            return 0.0
        else:
            score = 0.0
            for i, shortcode in enumerate(self.shortcodes):
                score += self.expected_return_indicator_value[i] * \
                    self.benchmark_allocation.get(shortcode, 0.0)
            return score

    def get_deviation_from_benchmark_allocation(self, weights):
        """
        Function to ccompute current allocation's deviation from the benchmark allocation
        """
        # TODO scale weights to 1 as benchmark allocations sum to 1
        if self.config.allocation_deviation_penalty == 0.0:
            return 0.0
        else:
            actual_allocation = dict.fromkeys(self.benchmark_allocation, 0.0)
            for i, shortcode in enumerate(self.shortcodes):
                actual_allocation[shortcode] = weights[i]
            return sum([(actual_allocation[x] - self.benchmark_allocation.get(x, 0))**2 for x in actual_allocation])

    def compute_allocation_on_day(self, day):
        '''
        It contains logic for each position sizing logic to implement to get allocation on a given day
        '''
        self.recompute_weights(day)

    def compute_allocation(self):
        '''
        It contains logic to compute allocation for a given date range
        '''
        self.counter = 0
        for given_date in list(self.expected_return_value.index):
            if self.counter >= 1:
                # print(given_date)
                self.is_live = True
                self.compute_allocation_on_day(given_date)
                self.weights_dict[given_date] = self.last_weights
            # print('weights_array', given_date, self.last_weights)

            self.counter += 1

    def recompute_weights(self, mode=Run_Mode.SIM, given_date=None):

        assert self.is_live

        if self.perform_common_signal_operations_start():
            return

        for i in range(len(self.sec_weights_info.secids)):
            self.expected_return_indicator_value[i], is_expected_return_indicator_ready = self.get_indicator_value(
                self.expected_return_indicators[i])

        self.expected_return_indicator_value = np.round(self.expected_return_indicator_value, 10)

        self.expected_return_indicator_value = self.transform_indicators(self.expected_return_indicator_value)
        self.expected_return_indicator_value = self.expected_return_indicator_value / sum(
            np.abs(self.expected_return_indicator_value)) if sum(np.abs(
                self.expected_return_indicator_value)) > 0 else self.expected_return_indicator_value

        if self.mode == Run_Mode.SIM:
            # filter row and columns for the securities which we have data
            sec_name_indexer = SecurityNameIndexer.GetUniqueInstance()
            shortcodes = [sec_name_indexer.get_shortcode_from_id(secid) for secid in self.sec_weights_info.secids]
            corr_mat = self.correlation_instance.get_correlation_df_to_use_on_date(shortcodes, self.watch.current_date)

            if corr_mat.dropna().empty:
                corr_mat = np.identity(len(shortcodes))
            else:
                corr_mat = np.asmatrix(corr_mat.loc[shortcodes][shortcodes])
        else:
            corr_mat = np.asmatrix(
                self.correlation_instance.get_correlation_df_to_use_on_date(self.secnames, given_date))

        if self.common_config.to_regularize:
            self.expected_return_indicator_value = self.pca_regularize_indicators(
                corr_mat, self.expected_return_indicator_value, self.risk_model_value)

        self.past_weights_optim = self.expected_return_indicator_value / self.risk_model_value
        TargetRiskManager.scale_weights_to_correlation_based_target_risk(
            self.past_weights_optim, self.sec_weights_info.secids, self.risk_model_value, corr_mat,
            self.common_config.target_risk * 0.9)

        # Using CVXPY and built-in ECOS solver
        self.cov_mat = np.array(corr_mat) * (np.outer(self.risk_model_value, self.risk_model_value))

        if self.last_weights is None:
            self.last_weights = np.zeros(self.past_weights_optim.shape[0])
        while self.last_weights.shape[0] < self.past_weights_optim.shape[0]:
            self.last_weights = np.append(self.last_weights, 0.0)

        self.initialize_other_constraints()
        self.objective_fn()

        # We added constraint names for debugging
        assert (len(self.constraint_names) == len(self.constraints))

        prob = cp.Problem(self.objective, self.constraints)
        try:
            prob.solve(solver=cp.ECOS)
        except cp.error.DCPError as dcp_err:
            for idx, this_constraint in enumerate(self.constraints):
                if not this_constraint.is_dcp():
                    print("Constraint id {} is not convex. It's name is {}".format(idx, self.constraint_names[idx]))
            print("Objective function is convex: {} \n".format(self.objective.is_dcp()))
            for ac in self.lower_allocations:
                print(ac.flag_array)
            for ac in self.upper_allocations:
                print(ac.flag_array)

            raise dcp_err

        self.last_weights = self.weights_variable.value


if __name__ == '__main__':
    config = {
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
        'max_turnover': 1000.0
        # 'allocation_constraints': [0.6, 'VTI', 'BND', 1.0]
    }
    import pandas as pd
    start_date = date(1995, 1, 1)
    end_date = date(2018, 4, 24)
    a = MeanVarianceCvxOpt(Run_Mode.TEST, config=config, start_date=start_date, end_date=end_date)
    allocation_df = pd.DataFrame(a.weights_dict, index=['VTI', 'BND']).T
    print(allocation_df)

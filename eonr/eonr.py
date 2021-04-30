# -*- coding: utf-8 -*-
'''**Calculates the economic optimum nitrogen rate and plots the results**

``EONR`` is a Python package for computing the economic optimum nitrogen
fertilizer rate using data from agronomic field trials under economic
conditions defined by the user (i.e., grain price and fertilizer cost).
It can be used for any crop (e.g., corn, wheat, potatoes, etc.), but the
current version only supports use of the quadratic-plateau piecewise
model.

**Therefore, use caution in making sure that a quadratic-plateau model is
appropriate for your application.** Future versions should add support for
other models (quadratic, spherical, etc.) that may improve the fit of
experimental yield response to nitrogen for other crops.

*Optional arguments when creating an instance of* ``EONR``:

Parameters:
    cost_n_fert (float, optional): Cost of N fertilizer (default: 0.50)
    cost_n_social (float, optional): Social cost of N fertilier (default: 0.00)
    costs_fixed (float, optional): Fixed costs on a per area basis (default:
        0.00)
    price_grain (float, optional): Price of grain (default: 3.00)
    col_n_app (str, optional): Column name pointing to the rate of applied N
        fertilizer data (default: 'rate_n_applied_lbac')
    col_yld (str, optional): Column name pointing to the grain yield data. This
        column is multiplied by price_grain to create the 'grtn' column in
        ``EONR.df_data`` (default: 'yld_grain_dry_buac')
    col_crop_nup (str, optional): Column name pointing to crop N uptake data
        (default: 'nup_total_lbac')
    col_n_avail (str, optional): Column name pointing to available soil N plus
        fertilizer  (default: 'soil_plus_fert_n_lbac')
    col_year (str, optional): Column name pointing to year (default: 'year')
    col_location (str, optional): Column name pointing to location (default:
        'location')
    col_time_n (str, optional): Column name pointing to nitrogen application
        timing (default: 'time_n')
    unit_currency (str, optional): String describing the curency unit (default:
        '$')
    unit_fert (str, optional): String describing the "fertilizer" unit
        (default: 'lbs')
    unit_grain (str, optional): String describing the "grain" unit (default:
        'bu')
    unit_area (str, optional): String descibing the "area" unit (default: 'ac')
    model (str, optional): Statistical model used to fit N rate response.
        *'quad_plateau'* = quadratic plateau; *'quadratic'* = quadratic;
        ``None`` = fits each of the preceding models, and uses the one with the
        highest R2 (default: 'quad_plateau').
    ci_level (float, optional): Confidence interval level to save in
        ``EONR.df_results`` and to display in the EONR plot; note that
        confidence intervals are calculated at many alpha levels, and we should
        choose from that list - should be one of [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        0.667, 0.7, 0.8, 0.9, 0.95, or 0.99] (default: 0.90)
    base_dir (str, optional): Base file directory when saving results (default:
        None)
    base_zero (``bool``, optional): Determines if gross return to N is
        expressed as an absolute value or relative to the yield/return at the
        zero rate. If base_zero is True, observed data for the zero nitrogen
        rate will be standardized by subtracting the y-intercept
        (:math:`\\beta_0`) from the 'grtn' column of ``EONR.df_data``.
        (default: True)
    print_out (``bool``, optional): Determines if "non-essential" results are
        printed in the Python console (default: False)

Requirements:
    The minimum data requirement to utilize this package is observed (or
    simulated) experimental data of agronomic yield response to nitrogen
    fertilizer. In other words, your experiment should have multiple nitrogen
    rate treatments, and you should have measured the yield for each
    experimental plot at the end of the season. Suitable experimental design
    for your particular experiment is always suggested (e.g., it should
    probably be replicated).

'''

import numpy as np
import os
import pandas as pd
import re
from scikits import bootstrap
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import OptimizeWarning
import uncertainties as unc
from uncertainties import unumpy as unp
import warnings

from eonr import Models
from eonr import Plotting_tools


class EONR(object):
    '''
    ``EONR`` is a Python tool for computing the optimum nitrogen rate and its
    confidence intervals from agricultural research data.
    '''
    def __init__(self,
                 cost_n_fert=0.5,
                 cost_n_social=0.0,
                 costs_fixed=0.0,
                 price_grain=3.00,
                 col_n_app='rate_n_applied_lbac',
                 col_yld='yld_grain_dry_buac',
                 col_crop_nup='nup_total_lbac',
                 col_n_avail='soil_plus_fert_n_lbac',
                 col_year='year', col_location='location', col_time_n='time_n',
                 unit_currency='$',
                 unit_fert='lbs', unit_grain='bu', unit_area='ac',
                 model='quad_plateau', ci_level=0.9, base_dir=None,
                 base_zero=True, print_out=False):
        self.df_data = None
        self.cost_n_fert = cost_n_fert
        self.cost_n_social = cost_n_social
        self.costs_fixed = costs_fixed
        self.price_grain = price_grain
        self.price_ratio = ((self.cost_n_fert + self.cost_n_social) /
                            self.price_grain)
        self.col_n_app = col_n_app
        self.col_yld = col_yld
        self.col_crop_nup = col_crop_nup
        self.col_n_avail = col_n_avail
        self.col_year = col_year
        self.col_location = col_location
        self.col_time_n = col_time_n
        self.unit_currency = unit_currency
        self.unit_fert = unit_fert
        self.unit_grain = unit_grain
        self.unit_area = unit_area
        self.unit_rtn = '{0} per {1}'.format(unit_currency, unit_area)
        self.unit_nrate = '{0} per {1}'.format(unit_fert, unit_area)
        self.model = model
        self.model_temp = model
        self.ci_level = ci_level
        self.base_dir = base_dir
        self.base_zero = base_zero
        self.print_out = print_out
        self.location = None
        self.year = None
        self.time_n = None
        self.onr_name = None
        self.onr_acr = None

        self.R = 0  # price_ratio to use when finding theta2
        self.coefs_grtn = {}
        self.coefs_grtn_primary = {}  # only used if self.base_zero is True
        self.coefs_nrtn = {}
        self.coefs_social = {}
        self.results_temp = {}

        self.mrtn = None
        self.eonr = None
        self.costs_at_onr = None
        self.df_ci = None
        self.df_ci_pdf = None
        self.df_delta_tstat = None
        self.df_der = None
        self.df_linspace = None

        self.fig_delta_tstat = None
        self.fig_derivative = None
        self.fig_eonr = None
        self.fig_tau = None
        self.ci_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.667, 0.7, 0.8, 0.9,
                        0.95, 0.99]
        self.alpha_list = [1 - xi for xi in self.ci_list]
        self.df_results = pd.DataFrame(columns=['price_grain', 'cost_n_fert',
                                                'cost_n_social', 'costs_fixed',
                                                'price_ratio',
                                                'unit_price_grain',
                                                'unit_cost_n',
                                                'location', 'year', 'time_n',
                                                'model', 'base_zero', 'eonr',
                                                'eonr_bias', 'R*',
                                                'costs_at_onr', 'ci_level',
                                                'ci_wald_l', 'ci_wald_u',
                                                'ci_pl_l', 'ci_pl_u',
                                                'ci_boot_l', 'ci_boot_u',
                                                'mrtn', 'grtn_r2_adj',
                                                'grtn_rmse',
                                                'grtn_max_y', 'grtn_crit_x',
                                                'grtn_y_int', 'scn_lin_r2',
                                                'scn_lin_rmse', 'scn_exp_r2',
                                                'scn_exp_rmse'])
        self.bootstrap_ci = None
        if self.unit_grain == 'kg' and self.unit_area == 'ha':
            self.metric = True
        elif self.unit_grain == 'lbs' and self.unit_area == 'ac':
            self.metric = False
        else:  # unknown
            self.metric = False

        if self.base_dir is not None:
            if not os.path.isdir(self.base_dir):
                os.makedirs(self.base_dir)
        else:
            self.base_dir = os.getcwd()
            folder_name = 'eonr_temp_output'
            self.base_dir = os.path.join(self.base_dir, folder_name)
            if not os.path.isdir(self.base_dir):
                os.makedirs(self.base_dir)
        self.base_dir = os.path.join(self.base_dir, 'trad_000')

        if self.cost_n_social > 0:
            self.onr_name = 'Socially'
            self.onr_acr = 'SONR'
        elif self.cost_n_fert > 0:
            self.onr_name = 'Economic'
            self.onr_acr = 'EONR'
        else:
            self.onr_name = 'Agronomic'
            self.onr_acr = 'AONR'

        self.models = Models(self)
        self.plotting_tools = Plotting_tools(self)

        self.plot_eonr.__func__.__doc__ = self.plotting_tools.plot_eonr.__doc__
        self.plot_tau.__func__.__doc__ = self.plotting_tools.plot_tau.__doc__
        self.plot_save.__func__.__doc__ = self.plotting_tools.plot_save.__doc__

    #  Following are the miscellaneous functions
    def _reset_temp(self):
        '''
        Resets temporary variables to be sure nothing carries through from a
        previous run
        '''
        self.results_temp = {'grtn_y_int': None,
                             'scn_lin_r2': None,
                             'scn_lin_rmse': None,
                             'scn_exp_r2': None,
                             'scn_exp_rmse': None}

    def _find_trial_details(self):
        '''
        Uses EONR.col_XXXXXXX to get year, location, and time_n from
        EONR.df_data
        '''
        df = self.df_data.copy()
        try:
            self.location = df.iloc[0][self.col_location]
        except KeyError:
            if self.location is not None:
                print('Was not able to infer "{0}" from EONR.df_data; '
                      '"{0}" is currently set to {1}. If this is not '
                      'correct, adjust prior to plotting using '
                      'EONR.set_column_names(col_location="your_loc_col_name")'
                      ' or EONR.set_trial_details({0}="your_location")'
                      ''.format('location', self.location))
            else:
                print('{0} is not currently set. You may want to set prior to '
                      'plotting using '
                      'EONR.set_column_names(col_location="your_loc_col_name")'
                      ' or EONR.set_trial_details({0}="your_location")'
                      ''.format('location'))
        try:
            self.year = df.iloc[0][self.col_year]
        except KeyError:
            if self.year is not None:
                print('Was not able to infer "{0}" from EONR.df_data; '
                      '"{0}" is currently set to {1}. If this is not '
                      'correct, adjust prior to plotting using '
                      'EONR.set_column_names(col_year="your_year_col_name")'
                      ' or EONR.set_trial_details({0}="your_year")'
                      ''.format('year', self.year))
            else:
                print('{0} is not currently set. You may want to set prior to '
                      'plotting using '
                      'EONR.set_column_names(col_year="your_year_col_name")'
                      ' or EONR.set_trial_details({0}="your_year")'
                      ''.format('year'))
        try:
            self.time_n = df.iloc[0][self.col_time_n]
        except KeyError:
            if self.time_n is not None:
                print('Was not able to infer "{0}" from EONR.df_data; '
                      '"{0}" is currently set to {1}. If this is not '
                      'correct, adjust prior to plotting using '
                      'EONR.set_column_names(col_time_n="your_time_n_col_name")'
                      ' or EONR.set_trial_details({0}="your_time_n")'
                      ''.format('time_n', self.time_n))
            else:
                print('{0} is not currently set. You may want to set prior to '
                      'plotting using '
                      'EONR.set_column_names(col_time_n="your_time_n_col_name")'
                      ' or EONR.set_trial_details({0}="your_time_n")'
                      ''.format('time_n'))

    def _set_df(self, df):
        '''
        Basic setup for dataframe
        '''
        self.df_data = df.copy()
        self._find_trial_details()
        print('\nComputing {0} for {1} {2} {3}'
              '\nCost of N fertilizer: {4}{5:.2f} per {6}'
              '\nPrice grain: {4}{7:.2f} per {8}'
              '\nFixed costs: {4}{9:.2f} per {10}'
              ''.format(self.onr_acr, self.location, self.year, self.time_n,
                        self.unit_currency, self.cost_n_fert, self.unit_fert,
                        self.price_grain, self.unit_grain,
                        self.costs_fixed, self.unit_area))
        if self.cost_n_social > 0:
            print('Social cost of N: {0}{1:.2f} per {2}'
                  ''.format(self.unit_currency, self.cost_n_social,
                            self.unit_fert))

    def _replace_missing_vals(self, missing_val='.'):
        '''
        Finds missing data in pandas dataframe and replaces with np.nan
        '''
        df = self.df_data.copy()
        for row in df.index:
            for col in df.columns:
                if df.at[row, col] == missing_val:
                    df.at[row, col] = np.nan
        self.df_data = df.copy()

    #  Following are curve_fit helpers and statistical calculations
    def _best_fit_lin(self, col_x, col_y):
        '''
        Computes slope, intercept, r2, and RMSE of linear best fit line between
        <col_x> and <col_y> of eonr.df_data

        Note that the 'social_cost_n' (<col_y>) already considers the economic
        scenario, so it doesn't have to be added again later
        '''
        df = self.df_data.copy()
        X = df[col_x].values.reshape(-1)
        y = df[col_y].values.reshape(-1)
        mx, b, r_value, p_value, std_err = stats.linregress(X, y)
        lin_r2 = r_value**2
        res = y - (b + mx*X)
        ss_res = np.sum(res**2)
        lin_rmse = (ss_res)**0.5
        self.coefs_social['lin_mx'] = mx
        self.coefs_social['lin_b'] = b
        self.coefs_social['lin_r2'] = lin_r2
        self.coefs_social['lin_rmse'] = lin_rmse
        if self.print_out is True:
            print('\ny = {0:.5} + {1:.5}x'.format(b, mx))
            print('lin_r2 = {0:.3}'.format(lin_r2))
            print('RMSE = {0:.1f}\n'.format(lin_rmse))

    def _best_fit_exp(self, col_x, col_y, guess_a=10, guess_b=0.0001,
                      guess_c=-10):
        '''
        Computes a, b, and c of best fit exponential function between <col_x>
        and <col_y>

        Note that the 'social_cost_n' (<col_y>) already considers the economic
        scenario, so it doesn't have to be added again later
        '''
        df = self.df_data.copy()
        x = df[col_x].values.reshape(-1)
        y = df[col_y].values.reshape(-1)
        popt = None
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_best_fit_exp() -> _f_exp', col_x, col_y))
        try:
            popt, pcov = self._curve_fit_opt(self.models.exp, x, y,
                                             p0=(guess_a, guess_b, guess_c),
                                             maxfev=1500, info=info)
        except RuntimeError as err:
            print('\n{0}\nTrying a new guess before giving up..'.format(err))
            pass

        if popt is None:
            try:
                popt, pcov = self._curve_fit_opt(self.models.exp, x, y,
                                                 p0=(guess_a*10,
                                                     guess_b**10,
                                                     guess_c*10),
                                                 info=info)
            except RuntimeError as err:
                print('\n{0}\nTry adjusting the initial guess parameters..'
                      ''.format(err))
                pass
            except np.linalg.LinAlgError as err:
                print('\n{0}'.format(err))
                pass
        if popt is None or np.any(popt == np.inf):
            if self.print_out is True:
                print("Couldn't fit data to an exponential function..\n")
            self.coefs_social['exp_gamma0'] = None
            self.coefs_social['exp_gamma1'] = None
            self.coefs_social['exp_gamma2'] = None
            self.coefs_social['exp_r2'] = 0
            self.coefs_social['exp_rmse'] = None
        else:
            exp_r2, _, _, _, exp_rmse = self._get_rsq(self.models.exp, x, y,
                                                      popt)
            gamma0, gamma1, gamma2 = popt
            if self.print_out is True:
                print('y = {0:.5} * exp({1:.5}x) + {2:.5} '.format(gamma0, gamma1, gamma2))
                print('exp_r2 = {0:.3}'.format(exp_r2))
                print('RMSE = {0:.1f}\n'.format(exp_rmse))

            self.coefs_social['exp_gamma0'] = gamma0
            self.coefs_social['exp_gamma1'] = gamma1
            self.coefs_social['exp_gamma2'] = gamma2
            self.coefs_social['exp_r2'] = exp_r2
            self.coefs_social['exp_rmse'] = exp_rmse

    def _calc_aic(self, x, y, dist='gamma'):
        '''
        Calculate the Akaike information criterion (AIC) using either a gamma
        (<dist>='gamma') or normal (<dist>='normal') distribution
        '''
        if dist == 'gamma':
            fitted_params = stats.gamma.fit(y)
            log_lik = np.sum(stats.gamma.logpdf(y, fitted_params[0],
                                                loc=fitted_params[1],
                                                scale=fitted_params[2]))
        elif dist == 'normal':
            fitted_params = stats.norm.fit(y)
            log_lik = np.sum(stats.norm.logpdf(y, fitted_params[0],
                                               loc=fitted_params[1],
                                               scale=fitted_params[2]))
        k = len(fitted_params)
        aic = 2 * k - 2 * log_lik
        return aic

    def _curve_fit_bs(self, f, xdata, ydata, p0=None, maxfev=800):
        '''
        Helper function to be suppress the OptimizeWarning. The bootstrap
        computation doesn't use the covariance matrix anyways (which is the
        main cause of the OptimizeWarning being thrown).
        (added so I can figure out what part of the code is causing it)
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            try:
                popt, pcov = curve_fit(f, xdata, ydata, p0=p0,
                                       maxfev=maxfev)
                return popt, pcov
            except OptimizeWarning:
                pass
            warnings.simplefilter('ignore', OptimizeWarning)  # hides warning
            popt, pcov = curve_fit(f, xdata, ydata, p0=p0, maxfev=maxfev)
        return popt, pcov

    def _curve_fit_opt(self, f, xdata, ydata, p0=None, maxfev=800, info=None):
        '''
        Helper function to suppress the OptimizeWarning. The bootstrap
        computation doesn't use the covariance matrix anyways (which is the
        main cause of the OptimizeWarning being thrown).
        (added so I can figure out what part of the code is causing it)
        <info> is a variable holding the column headings of the x and y data
        and is printed out to provide information to know which curve_fit
        functions are throwing the OptimizeWarning
        '''
        if info is None:
            popt, pcov = curve_fit(f, xdata, ydata, p0=p0, maxfev=maxfev)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                try:
                    popt, pcov = curve_fit(f, xdata, ydata, p0=p0,
                                           maxfev=maxfev)
                except OptimizeWarning:
                    if self.print_out is True:
                        print('Information for which the OptimizeWarning was '
                              'thrown:\n{0}'.format(info))
                warnings.simplefilter('ignore', OptimizeWarning)  # hides warn
                # essentially ignore warning and run anyway
                popt, pcov = curve_fit(f, xdata, ydata, p0=p0,
                                       maxfev=maxfev)
        return popt, pcov

    def _curve_fit_runtime(self, func, x, y, guess, maxfev=800, info=None):
        '''
        Helper function to run curve_fit() and watch for a RuntimeError. If we
        get a RuntimeError, then increase maxfev by 10x and try again.
        Sometimes this solves the problem of not being able to fit the
        function. If so, returns popt and pcov; if not, prints the error and
        returns None.
        '''
        popt = None
        try:
            popt, pcov = self._curve_fit_opt(func, x, y, p0=guess,
                                             maxfev=maxfev, info=info)
        except RuntimeError as err:
            print(err)
            maxfev *= 10
            print('Increasing the maximum number of calls to the function to '
                  '{0} before giving up.\n'.format(maxfev))
        if popt is None:
            try:
                popt, pcov = self._curve_fit_opt(func, x, y, p0=guess,
                                                 maxfev=maxfev, info=info)
            except RuntimeError as err:
                print(err)
                print('Was not able to fit data to the function.')
        if popt is not None:
            return popt, pcov
        else:
            return None, None

    def _compute_R(self, col_x, col_y, epsilon=1e-3, step_size=0.1):
        '''
        Given the true EONR with social cost considered, goal is to find the
        price ratio that will provide a sum of squares calculation for the true
        EONR
        '''
        df_data = self.df_data.copy()
        x = df_data[col_x].values
        y = df_data[col_y].values
        if self.cost_n_social > 0 and self.eonr is not None:
            b2 = self.coefs_grtn['b2'].n
            b1 = self.coefs_grtn['b1'].n
            self.R = b1 + (2 * b2 * self.eonr)  # as a starting point
            self.models.update_eonr(self)
            guess = (self.coefs_grtn['b0'].n,
                     self.eonr,
                     self.coefs_grtn['b2'].n)
            if self.model_temp == 'quadratic':
                f_model = self.models.quadratic
                f_model_theta2 = self.models.q_theta2
            elif self.model_temp == 'quad_plateau':
                f_model = self.models.quad_plateau
                f_model_theta2 = self.models.qp_theta2
            info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                    ''.format(f_model, col_x, col_y))
            popt, pcov = self._curve_fit_runtime(f_model_theta2, x, y, guess,
                                                 maxfev=800)
            dif = abs(popt[1] - self.eonr)
            count = 0
            while dif > epsilon:
                print('Using the optimize_R() algorithm')
                if count == 10:
                    epsilon *= 10
                elif count >= 100:
                    print('Could not converge R to fit the '
                          'EONR after {0} attemps. Fitted '
                          'model is within {1} {2} of the '
                          'computed EONR.'
                          ''.format(count, dif,
                                    self.unit_nrate))
                    break
                count += 1
                if popt[1] > self.eonr:
                    self.R += step_size  # increase R
                else:  # assumes R should aloways be positive
                    # go back one and increase at finer step
                    self.R -= step_size
                    step_size *= 0.1  # change step_size
                    self.R += step_size
                self.models.update_eonr(self)
                popt, pcov = self._curve_fit_runtime(f_model_theta2,
                                                     x, y, guess, maxfev=800)
                dif = abs(popt[1] - self.eonr)
            res = y - f_model_theta2(x, *popt)
            ss_res = np.sum(res**2)
            if (popt is None or np.any(popt == np.inf) or
                    np.any(pcov == np.inf)):
                b0 = unc.ufloat(popt[0], 0)
                theta2 = unc.ufloat(popt[1], 0)
                b2 = unc.ufloat(popt[2], 0)
            else:
                b0, theta2, b2 = unc.correlated_values(popt, pcov)
            info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                    ''.format(f_model, col_x,
                              col_y + ' (residuals)'))
#            func = self.models.quad_plateau
            popt, pcov = self._curve_fit_opt(lambda x, b1: f_model(
                    x, b0.n, b1, b2.n), x, y, info=info)
            popt = np.insert(popt, 0, b2.n)
            popt = np.insert(popt, 2, b0.n)
            f = np.poly1d(popt)
            result = minimize_scalar(-f)

            self.coefs_nrtn['b0'] = b0
            self.coefs_nrtn['theta2'] = result['x']
            self.coefs_nrtn['b2'] = b2
            self.coefs_nrtn['theta2_social'] = theta2
            self.coefs_nrtn['popt_social'] = [popt[2],
                                              theta2,
                                              popt[0]]
            self.coefs_nrtn['ss_res_social'] = ss_res
            self.coefs_nrtn['eonr_bias'] = theta2 - self.eonr

        elif self.cost_n_social == 0:
            self.R = self.price_ratio * self.price_grain
        else:
            assert self.eonr is not None, 'Please compute EONR'
        self.models.update_eonr(self)

    def _get_rsq(self, func, x, y, popt):
        '''
        Calculate the r-square (and ajusted r-square) of <y> for function
        <func> with <popt> parameters
        '''
        res = y - func(x, *popt)
        ss_res_mean = np.mean(res**2)
        rmse = (ss_res_mean)**0.5
        y_mean = np.mean(y)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y-y_mean)**2)
        r2 = 1-ss_res/ss_tot
        p = len(popt)
        n = len(x)
        r2_adj = 1-(1-r2)*(n-1)/(n-p-1)
        return r2, r2_adj, ss_res, ss_tot, rmse

    #  Following are higher level functions
    def _build_mrtn_lines(self, n_steps=None):
        '''
        Builds the Net Return to N (MRTN) line for plotting

        Parameters:
        n_steps (``int``): Number of nitrogen rates to be calculated to
            represent the GRTN curve (default: None - set dynamically as two
            steps per unit N).
        '''
        df = self.df_data.copy()
        x_min = float(df.loc[:, [self.col_n_app]].min(axis=0))
        x_max = float(df.loc[:, [self.col_n_app]].max(axis=0))
        if n_steps is None:
            n_steps = int((x_max - x_min)*2)
        eonr_social_n = 0

        x1, y_fert_n, x1a = self._setup_ncost_curve(x_min, x_max, n_steps)
        eonr_fert_n = self.eonr * self.cost_n_fert
        y_grtn = self._setup_grtn_curve(x1, x1a, n_steps)
        if self.cost_n_social > 0:
            y_social_n, eonr_social_n, _, _ = self._build_social_curve(
                    x1, fixed=False)
            rtn = y_grtn - (y_fert_n + y_social_n)
            self.linspace_cost_n_social = (x1, y_social_n)
        else:
            rtn = y_grtn - y_fert_n

        while len(y_grtn) != n_steps:
            if len(y_grtn) < n_steps:
                y_grtn = np.append(y_grtn, y_grtn[-1])
            else:
                y_grtn = y_grtn[:-1]
        rtn_der = np.insert(abs(np.diff(rtn)), 0, np.nan)
        rtn_der2 = np.insert(abs(np.diff(rtn_der)), 0, np.nan)
        df_linspace = pd.DataFrame({'x': x1,
                                    'cost_n_fert': y_fert_n,
                                    'grtn': y_grtn,
                                    'rtn': rtn,
                                    'rtn_der': rtn_der,
                                    'rtn_der2': rtn_der2})
        if self.cost_n_social > 0:
            df_linspace['cost_n_social'] = y_social_n
        self.df_linspace = df_linspace
        self.costs_at_onr = eonr_fert_n + eonr_social_n

    def _ci_pdf(self, run_n=None, n_steps=1000):
        '''
        Calculates the probability density function across all calculatd
        confidence interval levels

        Parameters:
            run_n (``int``): (default: ``None``)
        '''
        if run_n is None:
            df_ci = self.df_ci[self.df_ci['run_n'] ==
                               self.df_ci['run_n'].max()]
        else:
            df_ci = self.df_ci[self.df_ci['run_n'] == run_n]
        val_min = df_ci[df_ci['level'] == 0.99]['pl_l'].values[0]
        val_max = df_ci[df_ci['level'] == 0.99]['pl_u'].values[0]
        x_all = np.linspace(val_min, val_max, n_steps)

        level_list = list(df_ci['level'].unique())
        weights = np.zeros(x_all.shape)
        for level in level_list:
            if level != 0:
                pl_l = df_ci[df_ci['level'] == level]['pl_l'].values[0]
                pl_u = df_ci[df_ci['level'] == level]['pl_u'].values[0]
                weight_in = (level*100)
                weight_out = 100-weight_in  # 99 because 0 CI is excluded
#                idx_l = (np.abs(x_all - pl_l)).argmin()  # find nearest index
#                idx_u = (np.abs(x_all - pl_u)).argmin()
                dif_l = (x_all - pl_l)  # find index above
                dif_u = (pl_u - x_all)  # find index below
                idx_l = np.where(dif_l>0, dif_l, dif_l.max()).argmin()
                idx_u = np.where(dif_u>0, dif_u, dif_u.max()).argmin()

                unit_weight_in = weight_in / (idx_u - idx_l)
                unit_weight_out = weight_out / ((idx_l - x_all.argmin()) + (n_steps - idx_u))
                weights[:idx_l] += unit_weight_out  # add to weights
                weights[idx_u:] += unit_weight_out
                weights[idx_l:idx_u] += unit_weight_in
        df_ci_pdf = pd.DataFrame({'rate_n': x_all,
                                  'weights': weights})
        self.df_ci_pdf = df_ci_pdf

    def _rtn_derivative(self,):
        '''
        Calcuales the first derivative of the return to N curve

        Parameters:
            run_n (``int``): (default: ``None``)
        '''
        df_ci = self.df_ci[self.df_ci['run_n'] ==
                              self.df_ci['run_n'].max()].copy()
        pl_l = df_ci[df_ci['level'] == self.ci_level]['pl_l'].values[0]
        pl_u = df_ci[df_ci['level'] == self.ci_level]['pl_u'].values[0]
        df = self.df_linspace[['x', 'rtn_der', 'rtn_der2']].copy()
        df_trim = df[(df['x'] >= pl_l) & (df['x'] <= pl_u)]
        df_trim = df_trim.loc[~(df_trim == 0).any(axis=1)]
        der_max = df_trim['rtn_der'].iloc[-10:].max()
        df_trim = df_trim[(df_trim['rtn_der'] <= der_max) &
                          (df_trim['rtn_der2'] > 0.001)]
        df_der_left = df_trim[df_trim['x'] < self.eonr]
        df_der_right = df_trim[df_trim['x'] > self.eonr]
        slope_coef = (len(df_trim['x']) /
                      (df_trim['x'].max() - df_trim['x'].min()))
        try:
            slope_l, _, _, _, _ = stats.linregress(df_der_left['x'],
                                                   df_der_left['rtn_der'])
            self.coefs_nrtn['der_slope_lower'] = slope_l * slope_coef
        except ValueError:
            self.coefs_nrtn['der_slope_lower'] = np.nan

        try:
            slope_u, _, _, _, _ = stats.linregress(df_der_right['x'],
                                                   df_der_right['rtn_der'])
            self.coefs_nrtn['der_slope_upper'] = slope_u * slope_coef
        except ValueError:
            self.coefs_nrtn['der_slope_upper'] = np.nan

    def _build_social_curve(self, x1, fixed=False):
        '''
        Generates an array for the Social cost of N curve
        '''
        ci_l, ci_u = None, None
        if fixed is True:
            y_social_n = x1 * self.cost_n_social
            eonr_social_n = self.eonr * self.cost_n_social
        else:
            if self.coefs_social['lin_rmse'] < self.coefs_social['exp_rmse']:
                y_social_n = self.coefs_social['lin_b'] +\
                        (x1 * self.coefs_social['lin_mx'])
                eonr_social_n = self.coefs_social['lin_b'] +\
                        (self.eonr * self.coefs_social['lin_mx'])
            else:
                # x1_exp = self.coefs_social['exp_gamma0'] *\
                #         unp.exp(self.coefs_social['exp_gamma1'] * x1) +\
                #         self.coefs_social['exp_gamma2']
                x1_exp = self.models.exp(
                    x1, self.coefs_social['exp_gamma0'],
                    self.coefs_social['exp_gamma1'],
                    self.coefs_social['exp_gamma2'])
                y_social_n = unp.nominal_values(x1_exp)
                # eonr_social_n = self.coefs_social['exp_gamma0'] *\
                #         unp.exp(self.coefs_social['exp_gamma1'] * self.eonr) +\
                #         self.coefs_social['exp_gamma2']
                eonr_social_n = self.models.exp(
                    self.eonr, self.coefs_social['exp_gamma0'],
                    self.coefs_social['exp_gamma1'],
                    self.coefs_social['exp_gamma2'])
                std = unp.std_devs(x1_exp)
                ci_l = (y_social_n - 2 * std)
                ci_u = (y_social_n - 2 * std)
        return y_social_n, eonr_social_n, ci_l, ci_u

    def _calc_grtn(self):
        '''
        Computes Gross Return to N and saves in df_data under column heading of
        'grtn'
        '''
        self.df_data['grtn'] = self.df_data[self.col_yld]*self.price_grain
        self._fit_model(col_x=self.col_n_app, col_y='grtn')
#        if model == 'quad_plateau':
#            # Calculate the coefficients describing the quadratic plateau model
#            self._quad_plateau(col_x=self.col_n_app, col_y='grtn')
#        elif model == 'quadratic':
#            self.
#        elif model == 'lin_plateau':
#            self._r_lin_plateau(col_x=self.col_n_app, col_y='grtn')
#            self._r_confint(level=0.8)
#        else:
#            raise NotImplementedError('{0} model not implemented'
#                                      ''.format(model))
        self.results_temp['grtn_y_int'] = self.coefs_grtn['b0'].n

        if self.base_zero is True:
            self.df_data['grtn'] = (self.df_data['grtn'] -
                                    self.coefs_grtn['b0'].n)
            self._fit_model(col_x=self.col_n_app, col_y='grtn', rerun=True)
        self.models.update_eonr(self)

    def _calc_nrtn(self, col_x, col_y):
        '''
        Calculates the net return to N. If cost_n_social > 0,
        _f_qp_theta2 uses actual N uptake data to derive the absolute
        cost of excess (or net negative) fertilizer (see _best_fit_lin() and
        _best_fit_exp()). This cost is already in units of $ based on the
        economic scenario, but keep in mind that it will almost certainly have
        an intercept other than zero.

        For example, if more N is taken up than applied, there is a net
        negative use (-net_use); in terms of dollars, net_use can be divided
        by the social cost/price of N to get into units of $, which is a unit
        that can be used with the price ratio R.
        '''
        df_data = self.df_data.copy()
        x = df_data[col_x].values
        y = df_data[col_y].values

        if self.model_temp == 'quadratic':
#            f_model = self.models.quadratic
            f_model_theta2 = self.models.q_theta2
        elif self.model_temp == 'quad_plateau':
#            f_model = self.models.quad_plateau
            f_model_theta2 = self.models.qp_theta2
        guess = (self.coefs_grtn['b0'].n,
                 self.coefs_grtn['crit_x'],
                 self.coefs_grtn['b2'].n)
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format(f_model_theta2,
                          col_x, col_y))
        popt, pcov = self._curve_fit_opt(f_model_theta2, x, y,
                                         p0=guess, maxfev=1000, info=info)
        res = y - f_model_theta2(x, *popt)
        # if cost_n_social > 0, this will be dif than coefs_grtn['ss_res']
        ss_res = np.sum(res**2)

        if popt is None or np.any(popt == np.inf) or np.any(pcov == np.inf):
            b0 = unc.ufloat(popt[0], 0)
            theta2 = unc.ufloat(popt[1], 0)
            b2 = unc.ufloat(popt[2], 0)
        else:
            b0, theta2, b2 = unc.correlated_values(popt, pcov)

        self.coefs_nrtn = {
                'b0': b0,
                'theta2': theta2,
                'b2': b2,
                'popt': popt,
                'pcov': pcov,
                'ss_res': ss_res
                }

    def _calc_social_cost(self, col_x, col_y):
        '''
        Computes the slope and intercept for the model describing the added
        social cost of N
        '''
        self.df_data['resid_n'] = (self.df_data[self.col_n_avail] -
                                   self.df_data[self.col_crop_nup])
        self.df_data['social_cost_n'] = self.df_data['resid_n'] *\
            self.cost_n_social
        if self.print_out is True:
            print('Computing best-fit line between {0} and {1}..'
                  ''.format(col_x, col_y))
        self._best_fit_lin(col_x, col_y)
        self._best_fit_exp(col_x, col_y)
        self.results_temp['scn_lin_r2'] = self.coefs_social['lin_r2']
        self.results_temp['scn_lin_rmse'] = self.coefs_social['lin_rmse']
        self.results_temp['scn_exp_r2'] = self.coefs_social['exp_r2']
        self.results_temp['scn_exp_rmse'] = self.coefs_social['exp_rmse']

    def _print_grtn(self):
        '''
        Prints results of Gross Return to N calculation
        '''
        print('\nN Rate vs. Gross Return to N ({0} at ${1:.2f} per {2})'
              ''.format(self.unit_rtn, self.price_grain, self.unit_grain))
        print('y = {0:.5} + {1:.5}x + {2:.5}x^2'.format(
                self.coefs_grtn['b0'].n,
                self.coefs_grtn['b1'].n,
                self.coefs_grtn['b2'].n))
        print('Critical N Rate: {0:.4}'.format(self.coefs_grtn['crit_x']))
        print('Maximum Y (approximate): {0:.4}'.format(
                self.coefs_grtn['max_y']))
        print('Adjusted R2: {0:.3f}'.format(self.coefs_grtn['r2_adj']))
        print('RMSE: {0:.1f} {1}'.format(self.coefs_grtn['rmse'],
              self.unit_rtn))

    def _print_results(self):
        '''
        Prints results of Economic Optimum N Rate calculation
        '''
        last_run_n = self.df_ci['run_n'].max()
        df_ci_last_all = self.df_ci[self.df_ci['run_n'] == last_run_n]
        fert_l = df_ci_last_all[df_ci_last_all['level'] == 0.5]['pl_l'].values[0]
        fert_u = df_ci_last_all[df_ci_last_all['level'] == 0.5]['pl_u'].values[0]

        df_ci_last = self.df_ci[(self.df_ci['run_n'] == last_run_n) &
                                (self.df_ci['level'] == self.ci_level)]
        try:
            pl_l = df_ci_last['pl_l'].values[0]
            pl_u = df_ci_last['pl_u'].values[0]
            wald_l = df_ci_last['wald_l'].values[0]
            wald_u = df_ci_last['wald_u'].values[0]
            if self.bootstrap_ci is True:
                boot_l = df_ci_last['boot_l'].values[0]
                boot_u = df_ci_last['boot_u'].values[0]
        except TypeError as err:
            print(err)
        print('{0} optimum N rate ({1}): {2:.1f} {3} [{4:.1f}, '
              '{5:.1f}] ({6:.1f}% confidence)'
              ''.format(self.onr_name, self.onr_acr, self.eonr,
                        self.unit_nrate, pl_l, pl_u, self.ci_level*100))
        print('Maximum return to N (MRTN): {0}{1:.2f} per {2}'
              ''.format(self.unit_currency, self.mrtn, self.unit_area))
#        print('Acceptable range in recommended fertilizer rate for {0} {1} '
#              '{2}: {3:.0f} to {4:.0f} {5}\n'
#              ''.format(self.location, self.year, self.time_n, fert_l, fert_u,
#                        self.unit_nrate))

        if self.print_out is True:
            print('Profile likelihood confidence bounds (90%): [{0:.1f}, '
                  '{1:.1f}]'.format(pl_l, pl_u))
            print('Wald confidence bounds (90%): [{0:.1f}, {1:.1f}]'
                  ''.format(wald_l, wald_u))
            print('Bootstrapped confidence bounds (90%): [{0:.1f}, {1:.1f}]\n'
                  ''.format(boot_l, boot_u))

    def _fit_model(self, col_x, col_y, rerun=False):
        '''
        Fits the specified model (EONR.model); if EONR.model is None, fits both
        then uses the model with the highest R^2 hereafter

        <col_x (``str``): df column name for x axis
        <col_y (``str``): df column name for y axis
        '''
        df_data = self.df_data.copy()
        x = df_data[col_x].values
        y = df_data[col_y].values
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_fit_model() -> _f_quad_plateau', col_x, col_y))
        guess = self._get_guess_qp(rerun=rerun)
        #  TODO: Add a try/except to catch a bad guess.. or at least warn the
        # user that the guess is *extremely* sensitive
        if self.model is None and rerun is False:
            print('Checking quadratic and quadric-plateau models for best '
                  'fit..')
            model_q = self.models.quadratic
            model_qp = self.models.quad_plateau
            popt_q, pcov_q = self._curve_fit_opt(model_q, x, y,
                                                 p0=guess, info=info)
            _, r2_adj_q, _, _, rmse_q = self._get_rsq(
                model_q, x, y, popt_q)
            popt_qp, pcov_qp = self._curve_fit_opt(model_qp, x,
                                                   y,p0=guess, info=info)
            _, r2_adj_qp, _, _, rmse_qp = self._get_rsq(
                model_qp, x, y, popt_qp)
            print('Quadratic model r^2: {0:.2f}'.format(r2_adj_q))
            print('Quadratic-plateau model r^2: {0:.2f}'.format(r2_adj_qp))
            if r2_adj_q > r2_adj_qp:
                self.model_temp = 'quadratic'
#                model = self.models.quadratic
#                popt, pcov = popt_q, pcov_q
                print('Using the quadratic model..')
            else:
                self.model_temp = 'quad_plateau'
#                model = self.models.quad_plateau
#                popt, pcov = popt_qp, pcov_qp
                print('Using the quadratic-plateau model..')
        elif self.model is None and rerun is True:
            # Using self.model_temp because it was already determined
            pass
        else:
            self.model_temp = self.model
        if self.model_temp == 'quadratic':
            f_model = self.models.quadratic
        elif self.model_temp == 'quad_plateau':
            f_model = self.models.quad_plateau
        else:
            raise NotImplementedError('{0} model not implemented'
                                      ''.format(self.model))
        popt, pcov = self._curve_fit_opt(f_model, x, y, p0=guess, info=info)

        # the following should be made robust to dynamically find the starting values for a dataset..
#        print(guess)
#        print(popt, pcov)
#        if popt[0] < 100 and rerun is False:
#            guess = (100, guess[1], guess[2])
#        if popt[1] < 1 and rerun is False:
#            guess = (guess[0], guess[1] * 4, guess[2])
#        if popt[2] > 0 and rerun is False:
#            guess = (guess[0], guess[1], guess[2] * 5)
#        popt, pcov = self._curve_fit_opt(f_model, x, y, p0=guess, info=info)
#        print('\n\n')
#        print(guess)
#        print(popt, pcov)

        if popt is None or np.any(popt == np.inf) or np.any(pcov == np.inf):
            b0 = unc.ufloat(popt[0], 0)
            b1 = unc.ufloat(popt[1], 0)
            b2 = unc.ufloat(popt[2], 0)
        else:
            try:
                b0, b1, b2 = unc.correlated_values(popt, pcov)
            except np.linalg.LinAlgError:
                b0 = unc.ufloat(popt[0], 0)
                b1 = unc.ufloat(popt[1], 0)
                b2 = unc.ufloat(popt[2], 0)
        crit_x = -b1.n/(2*b2.n)
        max_y = f_model(crit_x, b0.n, b1.n, b2.n)
        r2, r2_adj, ss_res, ss_tot, rmse = self._get_rsq(
                f_model, x, y, popt)
        aic = self._calc_aic(x, y, dist='gamma')

        if rerun is False:
            self.coefs_grtn = {
                    'model': self.model_temp,
                    'b0': b0,
                    'b1': b1,
                    'b2': b2,
                    'pcov': pcov,
                    'pval_a': None,
                    'pval_b': None,
                    'pval_c': None,
                    'r2': r2,
                    'r2_adj': r2_adj,
                    'AIC': aic,
                    'BIC': None,
                    'max_y': max_y,
                    'crit_x': crit_x,
                    'ss_res': ss_res,
                    'ss_tot': ss_tot,
                    'rmse': rmse
                    }
            self.coefs_nrtn = {
                    'b0': b0,
                    'theta2': None,
                    'b2': b2,
                    'popt': None,
                    'pcov': None,
                    'ss_res': None,
                    'eonr_bias': None,
                    'theta2_social': None,
                    'popt_social': None,
                    'ss_res_social': None
                    }
        else:
            self.coefs_grtn_primary = self.coefs_grtn.copy()
            self.coefs_grtn = {}
            self.coefs_grtn = {
                    'model': self.model_temp,
                    'b0': b0,
                    'b1': b1,
                    'b2': b2,
                    'pcov': pcov,
                    'pval_a': None,
                    'pval_b': None,
                    'pval_c': None,
                    'r2': r2,
                    'r2_adj': r2_adj,
                    'AIC': aic,
                    'BIC': None,
                    'max_y': max_y,
                    'crit_x': crit_x,
                    'ss_res': ss_res,
                    'ss_tot': ss_tot,
                    'rmse': rmse
                    }

    def _setup_ncost_curve(self, x_min, x_max, n_steps):
        '''
        Generates an array for the N cost curve
        '''
        if self.coefs_grtn['crit_x'] >= x_max:
            num_thresh = n_steps
        else:
            num_thresh = int(n_steps * (self.coefs_grtn['crit_x'] / (x_max)))
        step_size = (x_max - x_min) / (n_steps-1)
        x1a, ss1 = np.linspace(x_min,
                               ((x_min + (num_thresh-1) * step_size)),
                               num=num_thresh, retstep=True)
        x1b, ss2 = np.linspace(((x_min + (num_thresh) * step_size)),
                               x_max,
                               num=n_steps-num_thresh,
                               retstep=True)
        x1 = np.concatenate((x1a, x1b))
        y_fert_n = (x1 * self.cost_n_fert) + self.costs_fixed
        return x1, y_fert_n, x1a  # x1a used in _setup_grtn_curve()

    def _setup_grtn_curve(self, x1, x1a, n_steps):
        '''
        Generates an array for GRTN curve
        '''
        y_max = (self.coefs_grtn['b0'].n +
                 (self.coefs_grtn['crit_x'] * self.coefs_grtn['b1'].n) +
                 (self.coefs_grtn['crit_x'] * self.coefs_grtn['crit_x'] *
                  self.coefs_grtn['b2'].n))
        # Find index where all x = max
        y_temp = (self.coefs_grtn['b0'].n +
                  (x1*self.coefs_grtn['b1'].n) +
                  (x1*x1*self.coefs_grtn['b2'].n))
        if self.coefs_grtn['model'] == 'quad_plateau':
            y_max_idx = np.argmax(y_temp)
            y2a = (self.coefs_grtn['b0'].n +
                   (x1a[:y_max_idx]*self.coefs_grtn['b1'].n) +
                   (x1a[:y_max_idx]*x1a[:y_max_idx]*self.coefs_grtn['b2'].n))
            if self.eonr <= self.df_data[self.col_n_app].max():
                y2b = np.linspace(y_max, y_max, num=n_steps-y_max_idx)
            else:  # EONR is past the point of available data, plot last val again
                last_pt = (self.coefs_grtn['b0'].n +
                           (x1a[-1]*self.coefs_grtn['b1'].n) +
                           (x1a[-1]*x1a[-1]*self.coefs_grtn['b2'].n))
                y2b = np.linspace(last_pt, last_pt, num=n_steps-y_max_idx)
            y_grtn = np.concatenate((y2a, y2b))
        else:
            y_grtn = y_temp.copy()

        # if necessary, modify y_grtn so it has correct number of values
        if len(y_grtn) < n_steps:
            while len(y_grtn) < n_steps:
                y_grtn = np.concatenate((y_grtn, np.array(([y_max]))))
        elif len(y_grtn) > n_steps:
            while len(y_grtn) > n_steps:
                y_grtn = y_grtn[:-1].copy()
        else:
            pass
        return y_grtn

    def _solve_eonr(self):
        '''
        Uses scipy.optimize to find the maximum value of the return curve
        '''
        f_eonr1 = np.poly1d([self.coefs_nrtn['b2'].n,
                            self.coefs_nrtn['theta2'].n,
                            self.coefs_nrtn['b0'].n])
        f_eonr2 = np.poly1d([self.coefs_grtn['b2'].n,
                            self.coefs_grtn['b1'].n,
                            self.coefs_grtn['b0'].n])
        if self.cost_n_social > 0:
#            if self.coefs_social['lin_r2'] > self.coefs_social['exp_r2']:
#                print('method 1')
#                # subtract only cost of fertilizer
#                first_order = self.cost_n_fert + self.coefs_social['lin_mx']
#                f_eonr1 = self._modify_poly1d(f_eonr1, 1,
#                                              f_eonr1.coef[1] - first_order)
#                f_eonr1 = self._modify_poly1d(f_eonr1, 0,
#                                              self.coefs_social['lin_b'])
#                result = minimize_scalar(-f_eonr1)
#                self.f_eonr = f_eonr1
#            else:  # add together the cost of fertilizer and cost of social N
#                print('method 2')
            x_max = self.df_data[self.col_n_app].max()
            result = minimize_scalar(self.models.combine_rtn_cost,
                                     bounds=[-100, x_max+100],
                                     method='bounded')
        else:
            first_order = self.coefs_grtn['b1'].n - self.cost_n_fert
            f_eonr2 = self._modify_poly1d(f_eonr2, 1, first_order)
            f_eonr2 = self._modify_poly1d(
                    f_eonr2, 0, self.coefs_grtn['b0'].n-self.costs_fixed)
            result = minimize_scalar(-f_eonr2)
        # theta2 is EOR (minimum) only if total cost of N increases linearly
        # at a first order with an intercept of zero..
        self.eonr = result['x']
        self.mrtn = -result['fun']

    def _theta2_error(self):
        '''
        Calculates a error between EONR and theta2 from _f_qp_theta2
        '''
        df = self.df_data.copy()
        x = df[self.col_n_app].values
        y = df['grtn'].values
        if self.model_temp == 'quadratic':
#            f_model = self.models.quadratic
            f_model_theta2 = self.models.q_theta2
        elif self.model_temp == 'quad_plateau':
#            f_model = self.models.quad_plateau
            f_model_theta2 = self.models.qp_theta2
        guess = (self.coefs_grtn['b0'].n,
                 self.eonr,
                 self.coefs_grtn['b2'].n)
        popt, pcov = self._curve_fit_opt(f_model_theta2, x, y, p0=guess,
                                         maxfev=800)
        self.coefs_nrtn['eonr_bias'] = popt[1] - self.eonr

    #  Following are functions used in calculating confidence intervals
    def _bs_statfunction(self, x, y):
        '''
        '''
        maxfev = 1000
        b0 = self.coefs_grtn['b0'].n
        b2 = self.coefs_grtn['b2'].n
        if self.model_temp == 'quadratic':
#            f_model = self.models.quadratic
            f_model_theta2 = self.models.q_theta2
        elif self.model_temp == 'quad_plateau':
#            f_model = self.models.quad_plateau
            f_model_theta2 = self.models.qp_theta2
        guess = (b0, self.eonr, b2)
#        y = self.models.quad_plateau(x, a, b, c) + res
#        try_n = 0
        popt = [None, None, None]
        try:
            popt, _ = self._curve_fit_bs(f_model_theta2, x, y,
                                         p0=guess, maxfev=maxfev)
        except RuntimeError as err:
            print(err)
            maxfev = 10000
            print('Increasing the maximum number of calls to the function to '
                  '{0} before giving up.\n'.format(maxfev))
        if popt[1] is None:
            try:
                popt, _ = self._curve_fit_bs(f_model_theta2, x,
                                             y, p0=guess, maxfev=maxfev)
            except RuntimeError as err:
                print(err)
        return popt[1]

    def _build_df_ci(self):
        '''
        Builds a template to store confidence intervals in a dataframe
        '''
        df_ci = pd.DataFrame(data=[[self.df_data.iloc[0][self.col_location],
                                    self.df_data.iloc[0][self.col_year],
                                    self.df_data.iloc[0][self.col_time_n],
                                    self.price_grain,
                                    self.cost_n_fert,
                                    self.cost_n_social,
                                    self.price_ratio,
                                    0, 0, 0, np.nan, np.nan, np.nan, np.nan,
                                    np.nan, np.nan,
                                    'N/A', 'N/A']],
                             columns=['location', 'year', 'time_n',
                                      'price_grain',
                                      'cost_n_fert', 'cost_n_social',
                                      'price_ratio', 'f_stat', 't_stat',
                                      'level', 'wald_l', 'wald_u',
                                      'pl_l', 'pl_u', 'boot_l', 'boot_u',
                                      'opt_method_l', 'opt_method_u'])
        return df_ci

    def _calc_sse_full(self, x, y):
        '''
        Calculates the sum of squares across the full set of parameters,
        solving for theta2
        '''
        if self.model_temp == 'quadratic':
#            f_model = self.models.quadratic
            f_model_theta2 = self.models.q_theta2
        elif self.model_temp == 'quad_plateau':
#            f_model = self.models.quad_plateau
            f_model_theta2 = self.models.qp_theta2
        guess = (self.coefs_grtn['b0'].n,
                 self.eonr,
                 self.coefs_grtn['b2'].n)
        col_x = None  # Perhaps we should keep in col_x/ytil curve_fit runs..?
        col_y = None
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format(f_model_theta2,
                          col_x, col_y))
        popt, pcov = self._curve_fit_opt(f_model_theta2, x, y, p0=guess,
                                         info=info)
        res = y - f_model_theta2(x, *popt)
        sse_full_theta2 = np.sum(res**2)
        return sse_full_theta2

    def _check_progress_pl(self, step_size, tau_temp, tau, f_stat,
                           step_size_start, tau_delta_flag, count):
        '''
        Checks progress of profile likelihood function and adjusts step
        size accordingly. Without this function, there is a chance some
        datasets will take millions of tries if the ci is wide
        '''
        print('tau_temp', tau_temp)
        print('tau', tau)
        print('f_stat', f_stat)
        tau_delta = tau_temp - tau
        progress = tau_temp / f_stat
        print('progress', progress)
        print('')
        if progress < 0.9 and tau_delta < 1e-3:
            step_size = step_size_start
            step_size *= 1000
            tau_delta_flag = True
        elif progress < 0.9 and tau_delta < 1e-2:
            step_size = step_size_start
            step_size *= 100
            tau_delta_flag = True
        elif progress < 0.95 and count > 100:
            step_size = step_size_start
            step_size *= 1000
            tau_delta_flag = True
        elif progress < 0.9:
            step_size = step_size_start
            step_size *= 10
            tau_delta_flag = True
        else:
            if tau_delta_flag is True and count < 100:
                step_size = step_size_start
            # else keep step size the same
        return step_size, tau_delta_flag

    def _compute_bootstrap(self, alpha=0.1, samples_boot=9999):
        '''
        Uses bootstrapping to estimate EONR based on the sampling distribution
        0) Initialize vector x with residuals - mean
        1) Sample with replacement from vector x (n = number of initial
             samples..)
        2) Calculate the percentile and "bias corrected and accelerated"
             bootstrapped CIs
        '''
        boot_ci = [np.nan, np.nan]
        x = self.df_data[self.col_n_app].values
        y = self.df_data['grtn'].values
        try:
            boot_ci = bootstrap.ci((x, y), statfunction=self._bs_statfunction,
                                   alpha=alpha, n_samples=samples_boot,
                                   method='bca')
        except TypeError:
            if not isinstance(alpha, list):
                print('Unable to compute bootstrap confidence intervals at '
                      'alpha = {0}'.format(alpha))
        return boot_ci

    def _compute_cis(self, col_x, col_y, bootstrap_ci=True,
                     delta_tstat=False, samples_boot=9999):
        '''
        Computes Wald, Profile Likelihood, and Bootstrap confidence intervals.
        '''
        alpha_list = self.alpha_list
        df_ci = self._build_df_ci()
        cols = df_ci.columns
        self.df_delta_tstat = None  # reset from any previous datasets

        if bootstrap_ci is True:
#            df_ci = self._run_bootstrap(df_ci, alpha_list, n_samples=9999)
            pctle_list = self._parse_alpha_list(alpha_list)
            boot_ci = self._compute_bootstrap(alpha=pctle_list,
                                              samples_boot=samples_boot)
        else:
            boot_ci = [None] * ((len(self.alpha_list) * 2))
#        else:
#            boot_ci = np.insert(boot_ci, [0], [np.nan, np.nan])
        if boot_ci is not None:
            df_boot = self._parse_boot_ci(boot_ci)

        for alpha in alpha_list:
            level = 1 - alpha
            f_stat = stats.f.ppf(1-alpha, dfn=1, dfd=len(self.df_data)-3)
            t_stat = stats.t.ppf(1-alpha/2, len(self.df_data)-3)
#            if level == self.ci_level:
            pl_l, pl_u, wald_l, wald_u, opt_method_l, opt_method_u =\
                    self._get_likelihood(alpha, col_x, col_y, stat='t',
                                         delta_tstat=delta_tstat)
#            else:
#                pl_l, pl_u, wald_l, wald_u, opt_method_l, opt_method_u =\
#                        self._get_likelihood(alpha, col_x, col_y, stat='t',
#                                             delta_tstat=False)
            if bootstrap_ci is True:
                if boot_ci is None:
                    pctle = self._parse_alpha_list(alpha)
                    boot_l, boot_u = self._compute_bootstrap(
                            alpha=pctle, samples_boot=samples_boot)
                else:
                    boot_l = df_boot[df_boot['alpha']==alpha]['boot_l'].values[0]
                    boot_u = df_boot[df_boot['alpha']==alpha]['boot_u'].values[0]
            else:
                boot_l, boot_u = np.nan, np.nan
            df_row = pd.DataFrame([[self.df_data.iloc[0][self.col_location],
                                    self.df_data.iloc[0][self.col_year],
                                    self.df_data.iloc[0][self.col_time_n],
                                    self.price_grain,
                                    self.cost_n_fert,
                                    self.cost_n_social,
                                    self.price_ratio,
                                    f_stat, t_stat, level,
                                    wald_l, wald_u,
                                    pl_l, pl_u, boot_l, boot_u,
                                    opt_method_l, opt_method_u]],
                                  columns=cols)
            df_ci = df_ci.append(df_row, ignore_index=True)


#            if df_row['level'].values[0] == self.ci_level:
#                df_ci_last = df_row
#        if bootstrap_ci is True:
##            df_ci = self._run_bootstrap(df_ci, alpha_list, n_samples=9999)
#            pctle_list = self._parse_alpha_list(alpha_list)
#            df_boot = self._run_bootstrap(pctle_list, n_samples=9999)
#            df_ci = pd.concat([df_ci, df_boot], axis=1)


        if self.df_ci is None:
            df_ci.insert(loc=0, column='run_n', value=1)
            self.df_ci = df_ci
        else:
            last_run_n = self.df_ci.iloc[-1, :]['run_n']
            df_ci.insert(loc=0, column='run_n', value=last_run_n+1)
            self.df_ci = self.df_ci.append(df_ci, ignore_index=True)
        last_run_n = self.df_ci.iloc[-1, :]['run_n']
#        self.df_ci_last = self.df_ci[(self.df_ci['run_n'] == last_run_n) &
#                                     (self.df_ci['level'] == self.ci_level)]

    def _compute_residuals(self):
        '''
        Computes the residuals of the gross return values and saves to
        <df_data> for use in the _compute_cis() function (confidence intervals)
        '''
        col_x = self.col_n_app
        col_y = 'grtn'
        df_data = self.df_data.copy()
        x = df_data[col_x].values
        y = df_data[col_y].values
        if self.model_temp == 'quadratic':
            f_model = self.models.quadratic
        elif self.model_temp == 'quad_plateau':
            f_model = self.models.quad_plateau
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format(f_model, col_x, col_y))
        guess = (self.coefs_grtn['b0'].n, self.coefs_grtn['b1'].n,
                 self.coefs_grtn['b2'].n)
#        print('popt: {0}'.format(guess))
        popt, pcov = self._curve_fit_opt(f_model, x, y, p0=guess, info=info)
#        if self.base_zero is False:
#            popt, pcov = self._curve_fit_opt(f_model, x, y,
#                                             p0=(600, 3, -0.01), info=info)
#        else:
#            popt, pcov = self._curve_fit_opt(f_model, x, y,
#                                             p0=(0, 3, -0.01), info=info)
#        print('popt: {0}'.format(popt))  # if same, do we need previous 4 lines?
#        res = y - self.models.quad_plateau(x, *popt)
        res = y - f_model(x, *popt)
        res -= res.mean()
        df_temp = pd.DataFrame(data=res, index=df_data.index,
                               columns=['grtn_res'])
        df_data = pd.concat([df_data, df_temp], axis=1)
        self.df_data = df_data

    def _compute_wald(self, n, p, alpha, s2c=None):
        '''
        Computes the Wald confidence intervals for a range of alphas. <n> and
        <p> are used to determine tau (from the t-statistic)
        From page 104 - 105 (Gallant, 1987)

        <n (``int``): number of samples
        <p (``int``): number of parameters
        <s2c (``float``): the variance of the EONR value(notation from Gallant, 1987)

        s2 = SSE / (n - p)
        c = s2c / s2
        tau * (s2 * c)**(0.5)
        '''
        if s2c is None:
            if self.cost_n_social > 0:
                s2c = self.coefs_nrtn['theta2_social'].s**2
            else:
                s2c = self.coefs_nrtn['theta2'].s**2
        tau = stats.t.ppf(1-alpha/2, n-p)  # Wald should use t_stat
        ci_l = self.eonr - (tau * s2c**(0.5))
        ci_u = self.eonr + (tau * s2c**(0.5))
        return ci_l, ci_u

    def _get_guess_qp(self, rerun=False):
        '''
        Gets a reasonable guess for p0 of curve_fit function
        * Note that these guesses are *extremely* sensitive to the result that
        will be generated from the curve_fit() function. The parameter guesses
        are set realistically based on yield response to N in MN, but there is
        no guarantee it will work for all datasets..
        '''
        if rerun is False and self.metric is False:
            guess = (600, 3, -0.01)
        elif rerun is False and self.metric is True:
            guess = (900, 10, -0.03)
        # if rerun is True, don't we already know the beta1 and beta2 params?
        elif rerun is True:
            if self.base_zero is True:
                b0 = 0
            else:
                b0 = self.coefs_grtn['b0'].n
            b1 = self.coefs_grtn['b1'].n
            b2 = self.coefs_grtn['b2'].n
            guess = (b0, b1, b2)
        return guess

    class _pl_steps_init(object):
        '''Initializes variables required for _get_pl_steps()'''
        def __init__(self, theta2_start, alpha, x, y, side, tau_start,
                     step_size, guess, sse_full, col_x, col_y,
                     cost_n_social, **kwargs):
            self.theta2_start = theta2_start
            self.alpha = alpha
            self.x = x
            self.y = y
            self.side = side
            self.tau_start = tau_start
            self.step_size = step_size
            self.guess = guess
            self.sse_full = sse_full
#            self.col_x = col_x
#            self.col_y = col_y
            self.cost_n_social = cost_n_social
            self.__dict__.update(kwargs)

            msg = ('Please choose either "upper" or "lower" for <side> of '
                   'confidence interval to compute.')
            assert side.lower() in ['upper', 'lower'], msg

            self.q = 1
            self.n = len(x)
            self.p = len(guess)
            self.f_stat = stats.f.ppf(1-alpha, dfn=self.q, dfd=self.n-self.p)  # ppf is inv of cdf
            self.t_stat = stats.t.ppf(1-alpha/2, self.n-self.p)
#            f_stat = stats.f.ppf(1-alpha, dfn=q, dfd=n-p)
            self.s2 = sse_full / (self.n - self.p)
            self.theta2 = theta2_start
            self.tau = tau_start
            self.step_size_start = step_size
            self.tau_delta_flag = False
            self.stop_flag = False
            str_func = '_get_likelihood() -> _f_qp_theta2'
            self.info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                         ''.format(str_func, col_x, col_y))

    def _get_pl_steps(self, theta2_start, alpha, x, y, side, tau_start=0,
                      step_size=1, epsilon=1e-9, df=None, sse_full=None,
                      stat='t', count=0):
        '''
        Computes the profile-likelihood confidence values by incrementing by
        <step_size> until tau and test statistic are within <epsilon>
        <theta2_start>: starting point of theta2 (should be set to maximum
            likelihood value)
        <alpha>: the significance level to compute the likelihood for
        <x> and <y>: the x and y data that likelihood should be computed for
        <side>: The side of the confidence interval to compute the likelihood
            for - should be either "upper" or "lower" (dtype = str)

        Uses <alpha> to calculate the inverse of the cdf (cumulative
        distribution function) of the F statistic. The T statistic can be used
        as well (they will give the same result).
        '''
        if self.model_temp == 'quadratic':
#            f_model = self.models.quadratic
            f_model_theta2 = self.models.q_theta2
        elif self.model_temp == 'quad_plateau':
#            f_model = self.models.quad_plateau
            f_model_theta2 = self.models.qp_theta2
        guess = (self.coefs_grtn['b0'].n,
                 self.coefs_grtn['crit_x'],
                 self.coefs_grtn['b2'].n)
        if sse_full is None:
            sse_full = self._calc_sse_full(x, y)
        if df is None:
            df = pd.DataFrame(columns=['theta', 'f_stat', 't_stat'])
        col_x = self.col_n_app
        col_y = 'grtn'
        cost_n_social = self.cost_n_social
        # Second, call like_init() class to itialize the get_likelihood() func
        li = self._pl_steps_init(
                theta2_start=theta2_start, alpha=alpha, x=x, y=y, side=side,
                tau_start=tau_start, step_size=step_size, epsilon=epsilon,
                guess=guess, sse_full=sse_full, col_x=col_x,
                col_y=col_y, cost_n_social=cost_n_social)
        if stat == 't':
            crit_stat = li.t_stat
        else:
            crit_stat = li.f_stat
        # Third, minimize the difference between tau and the test statistic
        # call, anything in _pl_steps_init() using li.<variable>
        # e.g., li.tau will get you the <tau> variable
        # Rule of thumb: if saved in both EONR and _pl_steps_init, use
        # variable from EONR; if passed directly to _get_pl_steps(), use the
        # variable directly

        # Testing quadratic model 8/7/2019

#        b0 = my_eonr.coefs_grtn['b0'].n
#        b1 = my_eonr.coefs_grtn['b1'].n
#        b2 = my_eonr.coefs_grtn['b2'].n
#        theta2 = my_eonr.coefs_nrtn['theta2'].n
#
#        y1 = my_eonr.models.quadratic(x, b0, b1, b2)
#        y2 = my_eonr.models.q_theta2(x, b0, theta2, b2)
#
#        sns.scatterplot(x, y1)
#        sns.scatterplot(x, y2)

        while li.tau < crit_stat:
            popt, pcov = self._curve_fit_runtime(
                    lambda x, b0, b2: f_model_theta2(
                            x, b0, li.theta2, b2), x, y,
                    guess=(1, 1), maxfev=800, info=li.info)
            if popt is not None:
                popt = np.insert(popt, 1, li.theta2)
                res = y - f_model_theta2(x, *popt)
                sse_res = np.sum(res**2)
                tau_temp_f = ((sse_res - sse_full) / li.q) / li.s2
                with warnings.catch_warnings():
                    warnings.simplefilter('error', RuntimeWarning)
                    try:
                        tau_temp_t = tau_temp_f**0.5
                    except RuntimeWarning:
                        tau_temp_t = 1e-6  # when 0, we get an overflow error
                    warnings.simplefilter('ignore', RuntimeWarning)

#                    print(err)
#                tau_temp_t = tau_temp_f**0.5
                if stat == 't':  # Be SURE tau is compared to correct stat!
                    tau_temp = tau_temp_t
                else:
                    tau_temp = tau_temp_f
                # The following is used in Hernandez and Mulla (2008), but adds
                # an unnecessary complexity/confusion to testing significance
#                tau_temp = abs(li.theta2 - self.eonr)**0.5 * tau_temp_f**0.5
#            print(alpha)
#            print(tau_temp - crit_stat)
            if count >= 1000:
                print('{0:.1f}% {1} profile likelihood failed to converge '
                      'after {2} iterations. Stopping calculation and using '
                      '{3:.1f} {4}'.format((1-alpha)*100, side.capitalize(),
                                           count, li.theta2,
                                           self.unit_nrate))
                tau_temp_f = li.f_stat
                tau_temp_t = li.t_stat
                li.stop_flag = True
                theta_same = df['theta'].iloc[-1]
                df2 = pd.DataFrame([[theta_same, li.f_stat, li.t_stat]],
                                   columns=['theta', 'f_stat', 't_stat'])
                df = df.append(df2, ignore_index=True)
                break  # break out of while loop
            elif popt is None:
                print('{0:.1f}% {1} profile likelihood failed to find optimal '
                      'parameters. Stopping calculation.'
                      ''.format((1-alpha)*100, side.capitalize()))
                tau_temp_f = li.f_stat
                tau_temp_t = li.t_stat
                li.stop_flag = True
                if df['theta'].iloc[-1] is not None:
                    theta_rec = df['theta'].iloc[-1]
                    if theta_rec < 1.0:
                        theta_rec = 0
                else:
                    theta_rec = np.NaN
                df2 = pd.DataFrame([[theta_rec,  li.f_stat, li.t_stat]],
                                   columns=['theta', 'f_stat', 't_stat'])
                df = df.append(df2, ignore_index=True)
                break  # break out of while loop
            else:
                df2 = pd.DataFrame([[li.theta2, tau_temp_f, tau_temp_t]],
                                   columns=['theta', 'f_stat', 't_stat'])
                df = df.append(df2, ignore_index=True)
            count += 1
#            if count > 3:
#                step_size, li.tau_delta_flag = self._check_progress_pl(
#                        step_size, tau_temp, li.tau, crit_stat,
#                        li.step_size_start, li.tau_delta_flag, count)
            if side.lower() == 'upper':
                li.theta2 += step_size
            elif side.lower() == 'lower':
                li.theta2 -= step_size
            li.tau = tau_temp
        if len(df) <= 1:
            # start over, reducing step size
            _, _, df = self._get_pl_steps(
                    theta2_start, alpha, x, y, side, tau_start=tau_start,
                    step_size=(step_size/10), epsilon=epsilon, df=df,
                    sse_full=sse_full, stat=stat, count=count)
        elif li.stop_flag is True:  # If we had to stop calculation..
            pass  # stop trying to compute profile-likelihood
#         if CI is moving faster than epsilon
        elif abs(df['theta'].iloc[-1] - df['theta'].iloc[-2]) > epsilon:
        # Can't stop when within x of epsilon because sometimes convergence
        # isn't reached
#        elif abs(tau_temp - crit_stat) > epsilon:
            df = df[:-1]
            # At this point, we could get stuck in a loop if we never make any
            # headway on closing in on epsilon
            if stat == 't':
                tau_start = df['t_stat'].iloc[-1]
            else:
                tau_start = df['f_stat'].iloc[-1]
            _, _, df = self._get_pl_steps(
                    df['theta'].iloc[-1], alpha, x, y, side,
                    tau_start=tau_start, step_size=(step_size/10),
                    epsilon=epsilon, df=df, sse_full=sse_full, stat=stat,
                    count=count)
        else:
            pass  #
#            boot_l, boot_u = self._compute_bootstrap(alpha, n_samples=5000)
#        theta2_out = df['theta'].iloc[-2:].mean()
#        tau_out = df['f_stat'].iloc[-2:].mean()
        theta2_out = df['theta'].iloc[-1]
        tau_out = df['t_stat'].iloc[-1]
        return theta2_out, tau_out, df

    def _run_minimize_pl(self, f, theta2_opt, pl_guess, method='Nelder-Mead',
                         side='lower', pl_guess_init=None):
        '''
        Runs the minimize function making sure the result is suitable/as
        expected for the profile likelihood
        '''
        if pl_guess_init is None:
            pl_guess_init = pl_guess
        if pl_guess <= 0:  # Should not be zero
            pl_guess = 0.5  # just guessing this is where it should be
        if side == 'lower':
            initial_guess = theta2_opt - pl_guess
        elif side == 'upper':
            initial_guess = theta2_opt + pl_guess
        result = minimize(f, initial_guess, method=method)
#            print(result)
        # print(pl_guess)
        if pl_guess > 800:
            pl_out = None
        elif result.success is not True:
            return self._run_minimize_pl(f, theta2_opt,
                                         pl_guess*1.05,
                                         method=method,
                                         side=side,
                                         pl_guess_init=pl_guess_init)
        elif result.success is True and side == 'lower':
#            if result.x[0] > self.eonr:
            if result.x[0] > theta2_opt:
                return self._run_minimize_pl(f, theta2_opt,
                                             pl_guess*1.05,
                                             method=method,
                                             side=side,
                                             pl_guess_init=pl_guess_init)
            else:
                pl_out = result.x[0]
        elif result.success is True and side == 'upper':
            if result.x[0] < theta2_opt:
                return self._run_minimize_pl(f, theta2_opt,
                                             pl_guess*1.05,
                                             method=method,
                                             side=side,
                                             pl_guess_init=pl_guess_init)
            else:
                pl_out = result.x[0]
#        else:  # finally, return result
#            pl_out = result.x[0]
        return pl_out

    def _get_likelihood(self, alpha, col_x, col_y, stat='t',
                        last_ci=[None, None], delta_tstat=False):
        '''
        Computes the profile liklihood confidence values using the sum of
        squares (see Gallant (1987), p. 107)
        <alpha>: the significance level to compute the likelihood for
        <x> and <y>: the x and y data that likelihood should computed for

        Uses <alpha> to calculate the inverse of the cdf (cumulative
        distribution function) of the F statistic. The T statistic can be used
        as well (they will give the same result).
        '''
        # First, initialize variables
        df = self.df_data.copy()
        x = df[col_x].values
        y = df[col_y].values
        guess = (self.coefs_grtn['b0'].n,
                 self.eonr,
                 self.coefs_grtn['b2'].n)
        sse_full = self._calc_sse_full(x, y)
        q = 1  # number of params being checked (held constant)
        n = len(x)
        p = len(guess)
        f_stat = stats.f.ppf(1-alpha, dfn=q, dfd=n-p)  # ppf is inv of cdf
        t_stat = stats.t.ppf(1-alpha/2, n-p)
        s2 = sse_full / (n - p)  # variance
        self.str_func = '_get_likelihood() -> _f_qp_theta2'
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format(self.str_func, col_x, col_y))
        # Second, minimize the difference between tau and the test statistic
        # call, anything in _get_likelihood_init() using li.<variable>
        # e.g., li.tau will get you the <tau> variable
        # Rule of thumb: if saved in both EONR and _get_likelihood_init, use
        # variable from EONR; if passed directly to _get_likelihood(), use the
        # variable directly

        def _f_like_opt(theta2):
            '''
            Function for scipy.optimize.newton() to optimize (find the minimum)
            of the difference between tau and the test statistic. This function
            returns <dif>, which will equal zero when the likelihood ratio is
            exactly equal to the test statistic (e.g., t-test or f-test)
            '''
            if self.model_temp == 'quadratic':
    #            f_model = self.models.quadratic
                f_model_theta2 = self.models.q_theta2
            elif self.model_temp == 'quad_plateau':
    #            f_model = self.models.quad_plateau
                f_model_theta2 = self.models.qp_theta2
            try:
                popt, pcov = self._curve_fit_runtime(
                        lambda x, b0, b2: f_model_theta2(
                                x, b0, theta2, b2), x, y, guess=(1, 1),
                        maxfev=800, info=info)
            except TypeError as e:
                popt = None
                pcov = None
                print('{0}\n{1}\nAlpha: {2}\n'.format(e, info, alpha))
            if popt is not None:
                popt = np.insert(popt, 1, theta2)
                res = y - f_model_theta2(x, *popt)
                sse_res = np.sum(res**2)
                tau_temp_f = ((sse_res - sse_full) / q) / s2
                with warnings.catch_warnings():
                    warnings.simplefilter('error', RuntimeWarning)
                    try:
                        tau_temp_t = tau_temp_f**0.5
                    except RuntimeWarning:
                        tau_temp_t = 1e-6  # when 0, we get an overflow error
                    warnings.simplefilter('ignore', RuntimeWarning)
                if stat == 't':  # Be SURE tau is compared to correct stat!
                    tau_temp = tau_temp_t
                    crit_stat = t_stat
                else:
                    tau_temp = tau_temp_f
                    crit_stat = f_stat
                dif = abs(crit_stat - tau_temp)
            elif popt is None:
                dif = None
            return dif

        def _delta_tstat(theta2_opt, pl_l, pl_u, alpha):
            '''
            Executes _f_like_opt() for a range of N rates/theta2 values
            '''
            if pl_l is np.nan:
                theta2_l = theta2_opt-100
            else:
                theta2_l = theta2_opt - (abs(theta2_opt-pl_l) * 1.1)
            if pl_u is np.nan:
                theta2_u = theta2_opt+100
            else:
                theta2_u = theta2_opt + (abs(theta2_opt-pl_u) * 1.1)
            theta2_hats = np.linspace(theta2_l, theta2_u, 400)
            dif_list = []
#            level_list = []
            for theta2_hat in theta2_hats:
                dif_list.append(_f_like_opt(theta2_hat))
#                level_list.append(1-alpha)

            df_delta_tstat = pd.DataFrame(data={'rate_n': theta2_hats,
                                                'delta_tstat': dif_list,
                                                'level': 1-alpha})
            return df_delta_tstat

        def _check_convergence(f, theta2_opt, pl_guess, pl_l, pl_u, alpha,
                               thresh=0.5, method='Nelder-Mead'):
            '''
            Check initial guess to see if delta_tau is close to zero - if not,
            redo with another intitial guess.

            Parameters
            thresh (float): tau(theta_2) threshold; anything greater than this
                is considered a poor fit, probably due to finding a local
                minimum.
            '''
            if pl_l is None:
                print('\nUpper {0:.2f} profile-likelihood CI may not have '
                      'optimized..'.format(alpha))
                pl_l = np.nan
            elif f(pl_l) > thresh:
                guess_l = theta2_opt - (pl_guess / 2)
                pl_l_reduced = minimize(f, guess_l, method=method)
                guess_l = theta2_opt - (pl_guess * 2)
                pl_l_increased = minimize(f, guess_l, method=method)
                if f(pl_l_reduced.x) < f(pl_l):
                    pl_l = pl_l_reduced.x[0]
                elif f(pl_l_increased.x) < f(pl_l):
                    pl_l = pl_l_increased.x[0]
                else:
                    print('\nLower {0:.2f} profile-likelihood CI may not have '
                          'optimized..'.format(alpha))
            if pl_u is None:
                print('\nUpper {0:.2f} profile-likelihood CI may not have '
                      'optimized..'.format(alpha))
                pl_u = np.nan
            elif f(pl_u) > thresh:
                guess_u = theta2_opt + (pl_guess / 2)
                pl_u_reduced = minimize(f, guess_u, method=method)
                guess_u = theta2_opt + (pl_guess * 2)
                pl_u_increased = minimize(f, guess_u, method=method)
                if f(pl_u_reduced.x) < f(pl_u):
                    pl_u = pl_u_reduced.x[0]
                elif f(pl_u_increased.x) < f(pl_u):
                    pl_u = pl_u_increased.x[0]
                else:
                    print('\nUpper {0:.2f} profile-likelihood CI may not have '
                          'optimized..'.format(alpha))
            return pl_l, pl_u

#        popt, pcov = self._curve_fit_opt(self._f_qp_theta2, x, y, p0=guess, maxfev=800, info=info)
        wald_l, wald_u = self._compute_wald(n, p, alpha)
        pl_guess = (wald_u - self.eonr)  # Adjust +/- init guess based on Wald
        theta2_bias = self.coefs_nrtn['eonr_bias']
        theta2_opt = self.eonr + theta2_bias  # check if this should add the 2

        # Lower CI: uses the Nelder-Mead algorithm
        method='Nelder-Mead'
        pl_l = self._run_minimize_pl(_f_like_opt, theta2_opt, pl_guess,
                                     method=method, side='lower')
        pl_u = self._run_minimize_pl(_f_like_opt, theta2_opt, pl_guess,
                                     method=method, side='upper')
        pl_l, pl_u = _check_convergence(_f_like_opt, theta2_opt, pl_guess,
                                        pl_l, pl_u, alpha, thresh=0.5,
                                        method=method)
#        if pl_l is not None:
#            pl_l += theta2_bias
#        if pl_u is not None:
#            pl_u += theta2_bias
#        if pl_l is None:
#            pl_l = np.nan
#        if pl_u is None:
#            pl_u = np.nan
        if pl_l > self.eonr or pl_u < self.eonr:  # can't trust the data
            print('Profile-likelihood calculations are not realistic: '
                  '[{0:.1f}, {1:.1f}] setting to NaN'.format(pl_l, pl_u))
            pl_l = np.nan
            pl_u = np.nan
            #  TODO: this will still add CIs to df_ci after the CI falls
            #  above/below the EONR. Could assume a uniform distribution and
            #  add this to theta2_bias for subsequent calculations. It seems
            #  as if theta2_bias changed from when it was set in coefs_nrtn:
            #  perhaps it should be recalculated
        if delta_tstat is True:
            df_temp = _delta_tstat(theta2_opt, pl_l, pl_u, alpha)
            if self.df_delta_tstat is None:
                self.df_delta_tstat = df_temp
            else:
                self.df_delta_tstat = self.df_delta_tstat.append(df_temp)
        return pl_l, pl_u, wald_l, wald_u, method, method

    def _handle_no_ci(self):
        '''
        If critical x is greater than the max N rate, don't calculate
        confidence intervals, but fill out df_ci with Wald CIs only
        '''
        df_ci = self._build_df_ci()
        guess = (self.coefs_grtn['b0'].n,
                 self.coefs_grtn['b1'].n,
                 self.coefs_grtn['b2'].n)
        for alpha in self.alpha_list:
            level = 1 - alpha
            n = len(self.df_data[self.col_n_app])
            p = len(guess)
            wald_l, wald_u = self._compute_wald(n, p, alpha)
            f_stat = stats.f.ppf(1-alpha, dfn=1, dfd=len(self.df_data)-3)
            t_stat = stats.t.ppf(1-alpha/2, len(self.df_data)-3)
            df_row = pd.DataFrame([[self.df_data.iloc[0][self.col_location],
                                    self.df_data.iloc[0][self.col_year],
                                    self.df_data.iloc[0][self.col_time_n],
                                    self.price_grain,
                                    self.cost_n_fert,
                                    self.cost_n_social,
                                    self.price_ratio,
                                    f_stat, t_stat, level,
                                    wald_l, wald_u, np.nan, np.nan,
                                    np.nan, np.nan, 'N/A', 'N/A']],
                                  columns=df_ci.columns)
            df_ci = df_ci.append(df_row, ignore_index=True)
#        boot_ci = [None] * ((len(self.ci_list) * 2))
#        boot_ci = [self.eonr, self.eonr] + list(boot_ci)
#        df_boot = self._parse_boot_ci(boot_ci)
#        df_ci = pd.concat([df_ci, df_boot], axis=1)
        if self.df_ci is None:
            df_ci.insert(loc=0, column='run_n', value=1)
            self.df_ci = df_ci
        else:
            last_run_n = self.df_ci.iloc[-1, :]['run_n']
            df_ci.insert(loc=0, column='run_n', value=last_run_n+1)
            self.df_ci = self.df_ci.append(df_ci, ignore_index=True)
        last_run_n = self.df_ci.iloc[-1, :]['run_n']
#        self.df_ci_last = self.df_ci[(self.df_ci['run_n'] == last_run_n) &
#                                     (self.df_ci['level'] == self.ci_level)]

    def _modify_poly1d(self, f, idx, new_val):
        '''
        Modifies a poly1d object, and returns the modifed object.
        '''
        assert idx in [0, 1, 2], 'Choose idx of 1, 2, or 3.'
        if idx == 2:
            f_new = np.poly1d([new_val, f[1], f[0]])
        elif idx == 1:
            f_new = np.poly1d([f[2], new_val, f[0]])
        elif idx == 0:
            f_new = np.poly1d([f[2],  f[1], new_val])
        return f_new

    def _parse_alpha_list(self, alpha):
        '''
        Creates a lower and upper percentile from a list of alpha values. The
        lower is (alpha / 2) and upper is (1 - (alpha / 2)). Required for
        scikits-bootstrap
        '''
        if isinstance(alpha, list):
            alpha_pctle = []
            for item in alpha:
                pctle_l = item / 2
                pctle_u = 1 - (item / 2)
                alpha_pctle.extend([pctle_l, pctle_u])
        else:
            pctle_l = alpha / 2
            pctle_u = 1 - (alpha / 2)
            alpha_pctle = [pctle_l, pctle_u]
        return alpha_pctle

    def _parse_boot_ci(self, boot_ci):
        '''
        Parses a list of values by separating into pairs and where the first
        item gets assigned to column 1 and the second items gets assigned to
        column 2. Returns a dataframe with both columns.
        '''
        def grouped(iterable, n):
            return zip(*[iter(iterable)]*n)

        boot_l = []
        boot_u = []
        for lower, upper in grouped(boot_ci, 2):
            boot_l.append(lower)
            boot_u.append(upper)
        df_boot = pd.DataFrame({'alpha': self.alpha_list,
                                'boot_l': boot_l,
                                'boot_u': boot_u})
        return df_boot

#    def _run_bootstrap(self, alpha, n_samples=9999):
#        '''
#        Calls the _compute_bootstrap() function.
#        '''
#        pctle = self._parse_alpha_list(alpha)
#        boot_ci = self._compute_bootstrap(alpha=pctle,
#                                          n_samples=n_samples)
##        if boot_ci is None:
##            boot_ci = []
##            for item in pctle_list:
##                boot_ci_temp = self._compute_bootstrap(alpha=item,
##                                                       n_samples=n_samples)
##                boot_ci.append(boot_ci_temp)
##        if boot_ci is None:
##            boot_ci = [None] * ((len(self.ci_list) * 2) + 2)
##        boot_ci = [self.eonr, self.eonr] + list(boot_ci)
##        df_boot = self._parse_boot_ci(boot_ci)
##        df_ci = pd.concat([df_ci, df_boot], axis=1)
#        return boot_ci

    def calc_delta(self, df_results=None):
        '''
        Calculates the change in EONR among economic scenarios.

        ``EONR.calc_delta`` filters all data by location, year, and
        nitrogen timing, then the "delta" is calculated as the difference
        relative to the economic scenario resulting in the highest EONR.

        Parameters:
            df_results (``Pandas dataframe``, optional): The dataframe
                containing the results from ``EONR.calculate_eonr()``
                (default: None).

        Returns:
            ``pandas.DataFrame``:
                **df_delta** -- The dataframe with the newly inserted EONR
                delta.


        Example:
            Please complete the `EONR.calculate_eonr`_ example first because
            this example builds on the results of the ``my_eonr`` object.

            Change the economic scenario (using ``EONR.calculate_eonr``) and
            calculate the EONR again for the same dataset (using
            ``EONR.calculate_eonr``)

            >>> price_grain = 0.314  # in USD per kg grain
            >>> my_eonr.update_econ(price_grain=price_grain)
            >>> my_eonr.calculate_eonr(df_data)
            Computing EONR for Minnesota 2012 Pre
            Cost of N fertilizer: $0.88 per kg
            Price grain: $0.31 per kg
            Fixed costs: $0.00 per ha
            Checking quadratic and quadric-plateau models for best fit..
            Quadratic model r^2: 0.72
            Quadratic-plateau model r^2: 0.73
            Using the quadratic-plateau model..
            Economic optimum N rate (EONR): 169.9 kg per ha [135.2, 220.9] (90.0% confidence)
            Maximum return to N (MRTN): $1682.04 per ha

            Use ``EONR.calc_delta`` to

            >>> df_delta = my_eonr.calc_delta(my_eonr.df_results)

            .. image:: ../img/calc_delta.png

        .. _EONR.calculate_eonr: eonr.EONR.html#eonr.EONR.calculate_eonr
        '''
        if df_results is None:
            df = self.df_results.unique()
        else:
            df = df_results.copy()

        years = df['year'].unique()
        years.sort()
        df_delta = None
        for year in years:
            df_year = df[df['year'] == year]
            locs = df_year['location'].unique()
            locs.sort()
            for loc in locs:
                df_loc = df_year[df_year['location'] == loc]
                times = df_loc[self.col_time_n].unique()
                for time in times:
                    df_yloct = df_loc[df_loc[self.col_time_n] == time]
                    eonr_base = df_yloct['eonr'].max()  # lowest fert:grain rat
                    eonr_delta = df_yloct['eonr'] - eonr_base
                    df_yloct.insert(8, 'eonr_delta', eonr_delta)
                    if df_delta is None:
                        df_delta = pd.DataFrame(columns=df_yloct.columns)
                    df_delta = df_delta.append(df_yloct)
        return df_delta

    def calculate_eonr(self, df, col_n_app=None, col_yld=None,
                       col_crop_nup=None, col_n_avail=None,
                       col_year=None, col_location=None, col_time_n=None,
                       bootstrap_ci=False, samples_boot=9999,
                       delta_tstat=False):
        '''
        Calculates the EONR and its confidence intervals.

        ``col_n_app`` and ``col_yld`` are required by ``EONR``, but not
        necessarily by ``EONR.calculate_eonr()``. They must either be set
        during the initialization of ``EONR``, before running
        ``EONR.calculate_eonr`` (using ``EONR.set_column_names``), or they
        must be passed in this ``EONR.calculate_eonr`` method.

        Parameters:
            df (``Pandas dataframe``): The dataframe containing the
                experimental data.
            col_n_app (``str``, optional): Column name pointing to the rate of
                applied N fertilizer data (default: None).
            col_yld (``str``, optional): Column name pointing to the grain
                yield data. This column is multiplied by price_grain to create
                the 'grtn' column in ``EONR.df_data`` (default: None).
            col_crop_nup (``str``, optional): Column name pointing to crop N
                uptake data (default: None).
            col_n_avail (``str``, optional): Column name pointing to available
                soil N at planting plus fertilizer throughout the season
                (default: None).
            col_year (``str``, optional): Column name pointing to year
                (default: None).
            col_location (``str``, optional): Column name pointing to location
                (default: None).
            col_time_n (``str``, optional): Column name pointing to nitrogen
                application timing (default: None).
            bootstrap_ci (``bool``, optional): Indicates whether bootstrap
                confidence intervals are to be computed. If calculating the
                EONR for many sites and/or economic scenarios, it may be
                desirable to set to ``False`` because the bootstrap confidence
                intervals take the most time to compute (default: False).
            samples_boot (``int``, optional): Number of samples in the
                bootstrap computation (default: 9999).
            delta_tstat (``bool``, optional): Indicates whether the
                difference from the t-statistic will be computed (as a function
                of theta2/N rate). May be useful to observe what optimization
                method is best suited to reach convergence when computing the
                profile-likelihood CIs (default: False).

        Note:
            ``col_crop_nup`` and ``col_n_avail`` are required to calculate the
            socially optimum nitrogen rate, SONR. The SONR is the optimum
            nitrogen rate considering the social cost of nitrogen, so
            therefore, ``EONR.cost_n_social`` must also be set. ``col_year``,
            ``col_location``, and ``col_time_n`` are purely  optional. They
            only affect the titles and axes labels of the plots.

        Example:
            Load and initialize ``eonr``

            >>> from eonr import EONR
            >>> import os
            >>> import pandas as pd

            Load the sample data

            >>> base_dir = r'F:\\nigo0024\Documents\GitHub\eonr\eonr'
            >>> df_data = pd.read_csv(os.path.join(base_dir, 'data', 'minnesota_2012.csv'))

            Set column names

            >>> col_n_app = 'rate_n_applied_kgha'
            >>> col_yld = 'yld_grain_dry_kgha'

            Set units

            >>> unit_currency = '$'
            >>> unit_fert = 'kg'
            >>> unit_grain = 'kg'
            >>> unit_area = 'ha'

            Set economic conditions

            >>> cost_n_fert = 0.88  # in USD per kg nitrogen
            >>> price_grain = 0.157  # in USD per kg grain

            Initialize ``EONR``

            >>> my_eonr = EONR(cost_n_fert=cost_n_fert,
                               price_grain=price_grain,
                               col_n_app=col_n_app,
                               col_yld=col_yld,
                               unit_currency=unit_currency,
                               unit_grain=unit_grain,
                               unit_fert=unit_fert,
                               unit_area=unit_area,
                               model=None,
                               base_dir=base_dir)

            Calculate the economic optimum nitrogen rate using
            ``EONR.calculate_eonr``

            >>> my_eonr.calculate_eonr(df_data)
            Computing EONR for Minnesota 2012 Pre
            Cost of N fertilizer: $0.88 per kg
            Price grain: $0.16 per kg
            Fixed costs: $0.00 per ha
            Checking quadratic and quadric-plateau models for best fit..
            Quadratic model r^2: 0.72
            Quadratic-plateau model r^2: 0.73
            Using the quadratic-plateau model..
            Economic optimum N rate (EONR): 162.3 kg per ha [130.5, 207.8] (90.0% confidence)
            Maximum return to N (MRTN): $767.93 per ha
        '''
        msg = ('Please set EONR.price_grain > 0.')
        assert self.price_grain > 0, msg
        if col_n_app is not None:
            self.col_n_app = str(col_n_app)
        if col_yld is not None:
            self.col_yld = str(col_yld)
        if col_crop_nup is not None:
            self.col_crop_nup = str(col_crop_nup)
        if col_n_avail is not None:
            self.col_n_avail = str(col_n_avail)
        self.bootstrap_ci = bootstrap_ci
        self._reset_temp()
        self._set_df(df)
        self._replace_missing_vals(missing_val='.')
        self._calc_grtn()
        if self.cost_n_social > 0:
            self._calc_social_cost(col_x=self.col_n_avail,
                                   col_y='social_cost_n')
        self._calc_nrtn(col_x=self.col_n_app, col_y='grtn')
        self._solve_eonr()
        self._compute_R(col_x=self.col_n_app, col_y='grtn')  # models.update_eonr
        self._theta2_error()
        if self.eonr > self.df_data[self.col_n_app].max():
            print('\n{0} is past the point of available data, so confidence '
                  'bounds are not being computed'.format(self.onr_acr))
            self._handle_no_ci()
        else:
            self._compute_residuals()
            self._compute_cis(col_x=self.col_n_app, col_y='grtn',
                              bootstrap_ci=bootstrap_ci,
                              samples_boot=samples_boot,
                              delta_tstat=delta_tstat)
        self._build_mrtn_lines()
        if self.costs_at_onr != 0:
            self._rtn_derivative()
#        self._ci_pdf()
        if self.print_out is True:
            self._print_grtn()
        self._print_results()
        if self.base_zero is True:
            base_zero = self.coefs_grtn_primary['b0'].n
            grtn_y_int = self.coefs_grtn['b0'].n
        else:
            base_zero = np.nan
            grtn_y_int = self.coefs_grtn['b0'].n
        'unit_grain', 'unit_costs',
        unit_price_grain = self.unit_rtn
        unit_cost_n = '{0} per {1}'.format(self.unit_currency,
                                           self.unit_fert)
        last_run_n = self.df_ci['run_n'].max()
        df_ci_last = self.df_ci[(self.df_ci['run_n'] == last_run_n) &
                                (self.df_ci['level'] == self.ci_level)]
        results = [[self.price_grain, self.cost_n_fert, self.cost_n_social,
                    self.costs_fixed,
                    self.price_ratio, unit_price_grain, unit_cost_n,
                    self.location, self.year, self.time_n, self.model_temp,
                    base_zero, self.eonr,
                    self.coefs_nrtn['eonr_bias'],
                    self.R, self.costs_at_onr,
                    self.ci_level, df_ci_last['wald_l'].values[0],
                    df_ci_last['wald_u'].values[0],
                    df_ci_last['pl_l'].values[0],
                    df_ci_last['pl_u'].values[0],
                    df_ci_last['boot_l'].values[0],
                    df_ci_last['boot_u'].values[0],
                    self.mrtn, self.coefs_grtn['r2_adj'],
                    self.coefs_grtn['rmse'],
                    self.coefs_grtn['max_y'],
                    self.coefs_grtn['crit_x'],
                    grtn_y_int,
                    self.results_temp['scn_lin_r2'],
                    self.results_temp['scn_lin_rmse'],
                    self.results_temp['scn_exp_r2'],
                    self.results_temp['scn_exp_rmse']]]

        self.df_results = self.df_results.append(pd.DataFrame(
                results, columns=self.df_results.columns),
                ignore_index=True)

    def plot_delta_tstat(self, level_list=None, style='ggplot'):
        '''Plots the test statistic as a function nitrogen rate

        Parameters:
            level_list (``list``): The confidence levels to plot; should be a
                subset of items in EONR.ci_list (default: None).
            style (``str``, optional): The style of the plolt; can be any of
                the options supported by ``matplotlib``

        Example:
            Load and initialize ``eonr``, then load the sample data

            >>> from eonr import EONR
            >>> import os
            >>> import pandas as pd
            >>> base_dir = r'F:\\nigo0024\Documents\GitHub\eonr\eonr'
            >>> df_data = pd.read_csv(os.path.join(base_dir, 'data', 'minnesota_2012.csv'))

            Set column names, units, and economic conditions

            >>> col_n_app = 'rate_n_applied_kgha'
            >>> col_yld = 'yld_grain_dry_kgha'
            >>> unit_currency = '$'
            >>> unit_fert = 'kg'
            >>> unit_grain = 'kg'
            >>> unit_area = 'ha'
            >>> cost_n_fert = 0.88  # in USD per kg nitrogen
            >>> price_grain = 0.157  # in USD per kg grain

            Initialize ``EONR``

            >>> my_eonr = EONR(cost_n_fert=cost_n_fert, price_grain=price_grain,
                               col_n_app=col_n_app, col_yld=col_yld,
                               unit_currency=unit_currency, unit_grain=unit_grain,
                               unit_fert=unit_fert, unit_area=unit_area,
                               model=None, base_dir=base_dir)

            Calculate the economic optimum nitrogen rate using
            ``EONR.calculate_eonr``, being sure to set ``delta_stat`` to
            ``True``

            >>> my_eonr.calculate_eonr(df_data, delta_tstat=True)
            Computing EONR for Minnesota 2012 Pre
            Cost of N fertilizer: $0.88 per kg
            Price grain: $0.16 per kg
            Fixed costs: $0.00 per ha
            Checking quadratic and quadric-plateau models for best fit..
            Quadratic model r^2: 0.72
            Quadratic-plateau model r^2: 0.73
            Using the quadratic-plateau model..
            Economic optimum N rate (EONR): 162.3 kg per ha [130.5, 207.8] (90.0% confidence)
            Maximum return to N (MRTN): $767.93 per ha

            Plot the Delta t-stat plot using ``EONR.plot_delta_tstat``

            >>> my_eonr.plot_delta_tstat()

            .. image:: ../img/plot_delta_tstat.png
        '''
        if self.plotting_tools is None:
            self.plotting_tools = Plotting_tools(self)
        else:
            self.plotting_tools.update_eonr(self)
        self.plotting_tools.plot_delta_tstat(level_list=level_list,
                                             style=style)
        self.fig_delta_tstat = self.plotting_tools.fig_delta_tstat

    def plot_derivative(self, ci_type='profile-likelihood', ci_level=None,
                        style='ggplot'):
        '''
        Plots a zoomed up view of the ONR and the derivative

        Parameters:
            ci_type (str): Indicates which confidence interval type should be
                plotted. Options are 'wald', to plot the Wald
                CIs; 'profile-likelihood', to plot the profile-likelihood
                CIs; or 'bootstrap', to plot the bootstrap CIs (default:
                'profile-likelihood').
            ci_level (float): The confidence interval level to be plotted, and
                must be one of the values in EONR.ci_list. If None, uses the
                EONR.ci_level (default: None).
            level (``float``): The confidence levels to plot; should be a
                value from EONR.ci_list (default: 0.90).
            style (``str``, optional): The style of the plolt; can be any of
                the options supported by ``matplotlib``

        Example:
            Please complete the `EONR.calculate_eonr`_ example first because
            this example builds on the results of the ``my_eonr`` object.

            >>> my_eonr.plot_derivative()

            .. image:: ../img/plot_derivative.png

        .. _EONR.calculate_eonr: eonr.EONR.html#eonr.EONR.calculate_eonr
        '''
        if self.plotting_tools is None:
            self.plotting_tools = Plotting_tools(self)
        else:
            self.plotting_tools.update_eonr(self)
        self.plotting_tools.plot_derivative(ci_type=ci_type, ci_level=ci_level,
                                            style=style)
        self.fig_derivative = self.plotting_tools.fig_derivative

    def plot_eonr(self, ci_type='profile-likelihood', ci_level=None,
                  run_n=None, x_min=None, x_max=None, y_min=None, y_max=None,
                  show_model=True, style='ggplot'):
        '''
        Plots EONR, MRTN, GRTN, net return, and nitrogen cost.

        If left as ``None``, ``x_min``, ``x_max``, ``y_min``, and ``y_max``
        are set by ``Matplotlib``.

        Parameters:
            ci_type (``str``, optional): Indicates which confidence interval
                type should be plotted. Options are 'wald', to plot the Wald
                CIs; 'profile-likelihood', to plot the profile-likelihood
                CIs; or 'bootstrap', to plot the bootstrap CIs (default:
                'profile-likelihood').
            ci_level (``float``, optional): The confidence interval level to be
                plotted, and must be one of the values in EONR.ci_list. If
                ``None``, uses the
                ``EONR.ci_level`` (default: None).
            run_n (``int``, optional): NOT IMPLEMENTED. The run number to plot,
                as indicated in EONR.df_results; if None, uses the most recent,
                or maximum, run_n in EONR.df_results (default: None).
            x_min (``int``, optional): The minimum x-bounds of the plot
                (default: None)
            x_max (``int``, optional): The maximum x-bounds of the plot
                (default: None)
            y_min (``int``, optional): The minimum y-bounds of the plot
                (default: None)
            y_max (``int``, optional): The maximum y-bounds of the plot
                (default: None)
            show_model (str): Whether to display the type of fitted model in
                the helper legend (default: True).
            style (``str``, optional): The style of the plot; can be any of
                the options supported by `matplotlib`_ (default: 'ggplot').

        Example:
            Please complete the `EONR.calculate_eonr`_ example first because
            this example builds on the results of the ``my_eonr`` object.

            >>> my_eonr.plot_eonr(x_min=-5, x_max=300, y_min=-100, y_max=1400)

            .. image:: ../img/plot_eonr.png

        .. _matplotlib: https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
        .. _EONR.calculate_eonr: eonr.EONR.html#eonr.EONR.calculate_eonr
        '''
        if self.plotting_tools is None:
            self.plotting_tools = Plotting_tools(self)
        else:
            self.plotting_tools.update_eonr(self)
        self.plotting_tools.plot_eonr(ci_type=ci_type, ci_level=ci_level,
                                      run_n=run_n, x_min=x_min, x_max=x_max,
                                      y_min=y_min, y_max=y_max, style=style)
        self.fig_eonr = self.plotting_tools.fig_eonr

    def plot_modify_size(self, fig=None, plotsize_x=7, plotsize_y=4,
                         labelsize=11):
        '''
        Modifies the size of the last plot generated

        Parameters:
            fig (``Matplotlib Figure``, optional): Matplotlib figure to modify
                (default: None)
            plotsize_x (``float``, optional): Sets x size of plot in inches
                (default: 7)
            plotsize_y (``float``, optional): Sets y size of plot in inches
                (default: 4)
            labelsize (``float``, optional): Sets tick and label
                (defaulat: 11)

        Example:
            Please complete the `EONR.calculate_eonr`_ and
            `EONR.plot_eonr`_ examples first because this example builds on
            the results of the ``my_eonr.fig_eonr.fig`` object.

            >>> my_eonr.plot_modify_size(fig=my_eonr.fig_eonr.fig, plotsize_x=5, plotsize_y=3, labelsize=9)

            .. image:: ../img/plot_modify_size.png

        .. _EONR.calculate_eonr: eonr.EONR.html#eonr.EONR.calculate_eonr
        .. _EONR.plot_eonr: eonr.EONR.html#eonr.EONR.plot_eonr
        '''
        self.plotting_tools.modify_size(fig=fig, plotsize_x=plotsize_x,
                                        plotsize_y=plotsize_y,
                                        labelsize=labelsize)

    def plot_modify_title(self, title_text, g=None, size_font=12):
        '''
        Allows user to replace the title text

        Parameters:
            title_text (``str``): New title text
            g (``matplotlib.figure``): Matplotlib figure object to modify
                (default: None)
            size_font (``float``): Font size to use (default: 12)

        Example:
            Please complete the `EONR.calculate_eonr`_ and
            `EONR.plot_eonr`_ examples first because this example builds on
            the results of the ``my_eonr.fig_eonr.fig`` object.

            >>> my_eonr.plot_modify_title('Preplant N fertilizer - Stewart, MN 2012', g=my_eonr.fig_eonr.fig, size_font=15)

            .. image:: ../img/plot_modify_title.png

        .. _EONR.calculate_eonr: eonr.EONR.html#eonr.EONR.calculate_eonr
        .. _EONR.plot_eonr: eonr.EONR.html#eonr.EONR.plot_eonr
        '''
        self.plotting_tools.modify_title(title_text, g=g, size_font=size_font)

    def plot_save(self, fname=None, base_dir=None, fig=None, dpi=300):
        '''Saves a generated matplotlib figure to file

        Parameters:
            fname (``str``, optional): Filename to save plot to (default: None)
            base_dir (``str``, optional): Base file directory when saving
                results (default: None)
            fig (eonr.fig, optional): EONR figure object to save (default:
                None)
            dpi (``int``, optional): Resolution to save the figure to in dots
                per inch (default: 300)

        Example:
            Please complete the `EONR.calculate_eonr`_ and
            `EONR.plot_eonr`_ examples first because this example builds on
            the results of the ``my_eonr.fig_eonr.fig`` object.

            Set output filename

            >>> fname = r'F:\\nigo0024\Downloads\eonr_fig.png'

            Save the most recent figure

            >>> my_eonr.plot_save(fname)
            ``fig`` is None, so saving the current (most recent) figure.

            >>> os.path.isfile(fname)
            True

        .. _EONR.calculate_eonr: eonr.EONR.html#eonr.EONR.calculate_eonr
        .. _EONR.plot_eonr: eonr.EONR.html#eonr.EONR.plot_eonr
        '''
        self.plotting_tools.plot_save(fname=fname, base_dir=base_dir, fig=fig,
                                      dpi=dpi)

    def plot_tau(self, y_axis='t_stat', emphasis='profile-likelihood',
                 run_n=None, style='ggplot'):
        '''Plots the test statistic as a function nitrogen rate

        Parameters:
            y_axis (``str``, optional): Value to plot on the y-axis. Options
                are 't_stat', to plot the *T statistic*; 'f_stat', to plot the
                *F-statistic*; or 'level', to plot the *confidence level*;
                (default: 't_stat').
            emphasis (``str``, optional): Indicates which confidence interval
                type, if any, should be emphasized. Options are 'wald', to
                empahsize the Wald CIs;
                'profile-likelihood', to empahsize the profile-likelihood CIs;
                'bootstrap', to empahsize the bootstrap CIs; or
                ``None``, to empahsize no CI (default: 'profile-likelihood').
            run_n (``int``, optional): The run number to plot, as indicated in
                ``EONR.df_ci``; if ``None``, uses the most recent, or maximum,
                run_n in ``EONR.df_ci`` (default: None).
            style (``str``, optional): The style of the plolt; can be any of
                the options supported by ``matplotlib``

        Example:
            Please complete the `EONR.calculate_eonr`_ example first because
            this example builds on the results of the ``my_eonr`` object.

            >>> my_eonr.plot_tau()

            .. image:: ../img/plot_tau.png

        .. _EONR.calculate_eonr: eonr.EONR.html#eonr.EONR.calculate_eonr
        '''
        if self.plotting_tools is None:
            self.plotting_tools = Plotting_tools(self)
        else:
            self.plotting_tools.update_eonr(self)
        self.plotting_tools.plot_tau(y_axis=y_axis, emphasis=emphasis,
                                     run_n=run_n, style=style)
        self.fig_tau = self.plotting_tools.fig_tau

    def print_results(self):
        '''
        Prints the results of the optimum nitrogen rate computation

        Example:
            Please complete the `EONR.calculate_eonr`_ example first because
            this example builds on the results of the ``my_eonr`` object.

            >>> my_eonr.print_results()
            Economic optimum N rate (EONR): 162.3 kg per ha [130.5, 207.8] (90.0% confidence)
            Maximum return to N (MRTN): $767.93 per ha
        '''
        self._print_results()

    def set_column_names(self, col_n_app=None, col_yld=None, col_crop_nup=None,
                         col_n_avail=None, col_year=None,
                         col_location=None, col_time_n=None):
        '''
        Sets the column name(s) for ``EONR.df_data``

        If these descriptions are used as metadata in the input dataset, they
        are accessed for plotting purposes. These parameters do not affect the
        calculation of the EONR or its confidence intervals in any way.

        Parameters:
            col_n_app (``str``, optional): Column name pointing to the rate of
                applied N fertilizer data (default: None).
            col_yld (``str``, optional): Column name pointing to the grain
                yield data. This column is multiplied by price_grain to create
                the 'grtn' column in ``EONR.df_data`` (default: None).
            col_crop_nup (``str``, optional): Column name pointing to crop N
                uptake data (default: None).
            col_n_avail (``str``, optional): Column name pointing to available
                soil N at planting plus fertilizer throughout the season
                (default: None).
            col_year (``str``, optional): Column name pointing to year
                (default: None).
            col_location (``str``, optional): Column name pointing to location
                (default: None).
            col_time_n (``str``, optional): Column name pointing to nitrogen
                application timing (default: None).

        Example:
            Load and initialize ``eonr``

            >>> from eonr import EONR
            >>> import os
            >>> import pandas as pd
            >>> base_dir = r'F:\\nigo0024\Documents\GitHub\eonr\eonr'
            >>> my_eonr = EONR(model=None, base_dir=base_dir)

            Set the column names using ``EONR.set_column_names``

            >>> my_eonr.set_column_names(col_n_app='rate_n_applied_kgha', col_yld='yld_grain_dry_kgha')
            >>> print(my_eonr.col_n_app)
            >>> print(my_eonr.col_yld)
            rate_n_applied_kgha
            yld_grain_dry_kgha
        '''
        if col_n_app is not None:
            self.col_n_app = str(col_n_app)
        if col_yld is not None:
            self.col_yld = str(col_yld)
        if col_crop_nup is not None:
            self.col_crop_nup = str(col_crop_nup)
        if col_n_avail is not None:
            self.col_n_avail = str(col_n_avail)
        if col_year is not None:
            self.col_year = str(col_year)
        if col_location is not None:
            self.col_location = str(col_location)
        if col_time_n is not None:
            self.col_time_n = str(col_time_n)
        if self.df_data is not None:
            self._find_trial_details()  # Use new col_name(s) to update details

    def set_units(self, unit_currency=None, unit_fert=None, unit_grain=None,
                  unit_area=None):
        '''
        Sets the units data in ``EONR.df_data`` and for reporting

        Parameters:
            unit_currency (``str``, optional): Currency unit, e.g., "$"
                (default: None).
            unit_fert (``str``, optional): Fertilizer unit, e.g., "lbs"
                (default: None).
            unit_grain (``str``, optional): Grain unit, e.g., "bu" (default:
                None).
            unit_area (``str``, optional): Area unit, e.g., "ac" (default:
                None).

        Example:
            Load and initialize ``eonr``

            >>> from eonr import EONR
            >>> import os
            >>> import pandas as pd
            >>> base_dir = r'F:\\nigo0024\Documents\GitHub\eonr\eonr'
            >>> my_eonr = EONR(model=None, base_dir=base_dir)

            Set the units using ``EONR.set_units``

            >>> my_eonr.set_units(unit_currency='USD', unit_fert='kg', unit_grain='kg', unit_area='ha')
            >>> print(my_eonr.unit_currency)
            >>> print(my_eonr.unit_fert)
            >>> print(my_eonr.unit_grain)
            >>> print(my_eonr.unit_area)
            USD
            kg
            kg
            ha
        '''
        if unit_currency is not None:
            self.unit_currency = str(unit_currency)
        if unit_fert is not None:
            self.unit_fert = str(unit_fert)
        if unit_grain is not None:
            self.unit_grain = str(unit_grain)
        if unit_area is not None:
            self.unit_area = str(unit_area)

    def set_trial_details(self, year=None, location=None, n_timing=None):
        '''
        Sets the year, location, or nitrogen timing

        If these descriptions are used as metadata in the input dataset, they
        are accessed for plotting purposes. These parameters do not affect the
        calculation of the EONR or its confidence intervals in any way.

        Parameters:
            year (``str`` or ``int``, optional): Year of experimental trial
                (default: None)
            location (``str`` or ``int``, optional): Location of experimental
                trial (default: None)
            n_timing (``str`` or ``int``, optional): Nitrogen timing of
                experimental trial (default: None)

        Example:
            Load and initialize ``eonr``

            >>> from eonr import EONR
            >>> import os
            >>> import pandas as pd
            >>> base_dir = r'F:\\nigo0024\Documents\GitHub\eonr\eonr'
            >>> my_eonr = EONR(model=None, base_dir=base_dir)

            Set the trial details using ``EONR.set_trial_details``

            >>> my_eonr.set_trial_details(year=2019, location='St. Paul, MN', n_timing='At planting')
            >>> print(my_eonr.year)
            >>> print(my_eonr.location)
            >>> print(my_eonr.n_timing)
            2019
            St. Paul, MN
            At planting
        '''
        if year is not None:
            self.year = int(year)
        if location is not None:
            self.location = location
        if n_timing is not None:
            self.n_timing = n_timing

    def update_econ(self, cost_n_fert=None, cost_n_social=None,
                    costs_fixed=None, price_grain=None):
        '''
        Sets or resets the nitrogen fertilizer cost, social cost of nitrogen,
        fixed costs, and/or grain price.

        The price ratio is recomputed based on the passed information, then
        the the lowest level folder in the base directory is renamed/adjusted
        (``EONR.base_dir``) based on to the price ratio. The folder name is
        set according to the economic scenario (useful when running ``EONR``
        for many different economic scenarios then plotting and saving results
        for each scenario).

        Parameters:
            cost_n_fert (``float``, optional): Cost of nitrogen fertilizer
                (default: None).
            cost_n_social (``float``, optional): Cost of pollution caused by
                excess nitrogen (default: None).
            costs_fixed (float, optional): Fixed costs on a per area basis
            (default: None)
            price_grain (``float``, optional): Price of grain (default: None).

        Example:
            Load and initialize ``eonr``

            >>> from eonr import EONR
            >>> import os
            >>> import pandas as pd
            >>> base_dir = r'F:\\nigo0024\Documents\GitHub\eonr\eonr'
            >>> my_eonr = EONR(model=None, base_dir=base_dir)

            Set/update the cost of fertilizer and price of grain using
            ``EONR.update_econ``

            >>> my_eonr.update_econ(cost_n_fert=0.88, price_grain=0.157)
            >>> print(my_eonr.price_ratio)
            >>> print(my_eonr.base_dir)
            5.605095541
            F:\\nigo0024\Documents\GitHub\eonr\eonr\trad_5605

            Set/update the social cost of nitrogen, again using
            ``EONR.update_econ``

            >>> my_eonr.update_econ(cost_n_social=1.1)
            >>> print(my_eonr.price_ratio)
            >>> print(my_eonr.base_dir)
            12.61146496
            F:\\nigo0024\Documents\GitHub\eonr\eonr\social_12611_1100

        '''
        if cost_n_fert is not None:
            self.cost_n_fert = cost_n_fert  # in USD per lb
        if cost_n_social is not None:
            self.cost_n_social = cost_n_social  # in USD per lb lost
        if costs_fixed is not None:
            self.costs_fixed = costs_fixed  # in USD per lb lost
        if price_grain is not None:
            self.price_grain = price_grain  # in USD
        self.price_ratio = ((self.cost_n_fert + self.cost_n_social) /
                            self.price_grain)
        if self.base_dir is not None:
            if self.cost_n_social != 0 and self.metric is False:
                join_name = '{0}_{1:.3f}_{2:.3f}'.format(
                        'social', self.price_ratio, self.cost_n_social)
            elif self.cost_n_social != 0 and self.metric is True:
                join_name = '{0}_{1:.1f}_{2:.3f}'.format(
                        'social', self.price_ratio, self.cost_n_social)
            elif self.cost_n_social == 0 and self.metric is False:
                join_name = '{0}_{1:.3f}'.format('trad', self.price_ratio)
            elif self.cost_n_social == 0 and self.metric is True:
                join_name = '{0}_{1:.1f}'.format('trad', self.price_ratio)
            else:
                join_name = '{0}_{1:.3f}'.format('trad', self.price_ratio)
            join_name = re.sub(r'[.]', '', join_name)
            self.base_dir = os.path.join(os.path.split(self.base_dir)[0],
                                         join_name)
        if self.cost_n_social > 0:
            self.onr_name = 'Socially'
            self.onr_acr = 'SONR'
        elif self.cost_n_fert > 0:
            self.onr_name = 'Economic'
            self.onr_acr = 'EONR'
        else:
            self.onr_name = 'Agronomic'
            self.onr_acr = 'AONR'

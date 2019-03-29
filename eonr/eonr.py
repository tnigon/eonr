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
        *'quad_plateau'* = quadratic plateau; *'lin_plateau'* = linear plateau
        (default: 'quad_plateau')
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
    def __init__(self,
                 cost_n_fert=0.5,
                 cost_n_social=0,
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
        self.price_grain = price_grain
        self.price_ratio = round((cost_n_fert+cost_n_social) / price_grain, 3)
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
        self.df_ci = None

        self.fig_eonr = None
        self.fig_tau = None
        self.linspace_cost_n_fert = None
        self.linspace_cost_n_social = None
        self.linspace_qp = None
        self.linspace_rtn = None
        self.df_ci_temp = None
        self.ci_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.667, 0.7, 0.8, 0.9,
                        0.95, 0.99]
        self.alpha_list = [1 - xi for xi in self.ci_list]
        self.df_results = pd.DataFrame(columns=['price_grain', 'cost_n_fert',
                                                'cost_n_social', 'price_ratio',
                                                'location', 'year', 'time_n',
                                                'base_zero', 'eonr',
                                                'theta2_error', 'ci_level',
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
        print('\nComputing {0} for {1} {2} {3}\nCost of N fertilizer: '
              '{4}{5:.2f} per {6}\nPrice grain: {7}{8:.2f} per {9}'
              ''.format(self.onr_acr, self.location, self.year, self.time_n,
                        self.unit_currency, self.cost_n_fert, self.unit_fert,
                        self.unit_currency, self.price_grain, self.unit_grain))
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
        self.models.update_eonr(self)
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
            self.coefs_social['exp_a'] = None
            self.coefs_social['exp_b'] = None
            self.coefs_social['exp_c'] = None
            self.coefs_social['exp_r2'] = 0
            self.coefs_social['exp_rmse'] = None
        else:
            exp_r2, _, _, _, exp_rmse = self._get_rsq(self.models.exp, x, y,
                                                      popt)
            a, b, c = popt
            if self.print_out is True:
                print('y = {0:.5} * exp({1:.5}x) + {2:.5} '.format(a, b, c))
                print('exp_r2 = {0:.3}'.format(exp_r2))
                print('RMSE = {0:.1f}\n'.format(exp_rmse))

            self.coefs_social['exp_a'] = a
            self.coefs_social['exp_b'] = b
            self.coefs_social['exp_c'] = c
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
        self.R = 0
        if self.cost_n_social > 0 and self.eonr is not None:
            b2 = self.coefs_grtn['b2'].n
            b = self.coefs_grtn['b1'].n
            self.R = b + (2 * b2 * self.eonr)  # as a starting point
            self.models.update_eonr(self)
            guess = (self.coefs_grtn['b0'].n,
                     self.eonr,
#                     self.coefs_grtn['crit_x'],
                     self.coefs_grtn['b2'].n)
            popt, pcov = self._curve_fit_runtime(self.models.qp_theta2, x,
                                                 y, guess, maxfev=800)
            dif = abs(popt[1] - self.eonr)
            count = 0
            while dif > epsilon:
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
                popt, pcov = self._curve_fit_runtime(self.models.qp_theta2,
                                                     x, y, guess, maxfev=800)
                dif = abs(popt[1] - self.eonr)
            res = y - self.models.qp_theta2(x, *popt)
            ss_res = np.sum(res**2)
            if (popt is None or np.any(popt == np.inf) or
                    np.any(pcov == np.inf)):
                b0 = unc.ufloat(popt[0], 0)
                theta2 = unc.ufloat(popt[1], 0)
                b2 = unc.ufloat(popt[2], 0)
            else:
                b0, theta2, b2 = unc.correlated_values(popt, pcov)
            info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                    ''.format('_compute_R() -> _f_quad_plateau', col_x,
                              col_y + ' (residuals)'))
            func = self.models.quad_plateau
            popt, pcov = self._curve_fit_opt(lambda x, b1: func(
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
            self.coefs_nrtn['theta2_error'] = theta2 - self.eonr

        elif self.cost_n_social == 0:
            self.R = self.price_ratio * self.price_grain
        else:
            assert self.eonr is not None, 'Please compute EONR'
        R = self.R
        return R

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
    def _build_mrtn_lines(self, n_steps=100):
        '''
        Builds the Net Return to N (MRTN) line for plotting
        col_n_app    --> column heading for N applied (``str``)
        '''
        df = self.df_data.copy()
        x_min = float(df.loc[:, [self.col_n_app]].min(axis=0))
        x_max = float(df.loc[:, [self.col_n_app]].max(axis=0))

        x1, y_fert_n, x1a = self._setup_ncost_curve(x_min, x_max, n_steps)
        y_grtn = self._setup_grtn_curve(x1, x1a, n_steps)
        if self.cost_n_social > 0:
            y_social_n, _, _ = self._build_social_curve(x1, fixed=False)
            rtn = y_grtn - (y_fert_n + y_social_n)
            self.linspace_cost_n_social = (x1, y_social_n)
        else:
            rtn = y_grtn - y_fert_n

        while len(y_grtn) != n_steps:
            if len(y_grtn) < n_steps:
                y_grtn = np.append(y_grtn, y_grtn[-1])
            else:
                y_grtn = y_grtn[:-1]

        self.linspace_cost_n_fert = (x1, y_fert_n)  # N cost
        self.linspace_qp = (x1, y_grtn)  # quadratic plateau
        self.linspace_rtn = (x1, rtn)

    def _build_social_curve(self, x1, fixed=True):
        '''
        Generates an array for the Social cost of N curve
        '''
        ci_l, ci_u = None, None
        if fixed is True:
            y_social_n = x1 * self.cost_n_social
        else:
            if self.coefs_social['lin_r2'] > self.coefs_social['exp_r2']:
                y_social_n = self.coefs_social['lin_b'] +\
                        (x1 * self.coefs_social['lin_mx'])
            else:
                x1_exp = self.coefs_social['exp_a'] *\
                        unp.exp(self.coefs_social['exp_b'] * x1) +\
                        self.coefs_social['exp_c']
                y_social_n = unp.nominal_values(x1_exp)
                std = unp.std_devs(x1_exp)
                ci_l = (y_social_n - 2 * std)
                ci_u = (y_social_n - 2 * std)
        return y_social_n, ci_l, ci_u

    def _calc_grtn(self, model='quad_plateau'):
        '''
        Computes Gross Return to N and saves in df_data under column heading of
        'grtn'
        '''
        self.df_data['grtn'] = self.df_data[self.col_yld]*self.price_grain
        if model == 'quad_plateau':
            # Calculate the coefficients describing the quadratic plateau model
            self._quad_plateau(col_x=self.col_n_app, col_y='grtn')
        elif model == 'lin_plateau':
            self._r_lin_plateau(col_x=self.col_n_app, col_y='grtn')
            self._r_confint(level=0.8)
        else:
            raise NotImplementedError('{0} model not implemented'
                                      ''.format(model))
        self.results_temp['grtn_y_int'] = self.coefs_grtn['b0'].n

        if self.base_zero is True:
            self.df_data['grtn'] = (self.df_data['grtn'] -
                                    self.coefs_grtn['b0'].n)
            if model == 'quad_plateau':
                self._quad_plateau(col_x=self.col_n_app, col_y='grtn',
                                   rerun=True)
            elif model == 'lin_plateau':
                self._r_lin_plateau(col_x=self.col_n_app, col_y='grtn')
                self._r_confint(level=0.8)
            else:
                raise NotImplementedError('{0} model not implemented'
                                          ''.format(model))

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

        guess = (self.coefs_grtn['b0'].n,
                 self.coefs_grtn['crit_x'],
                 self.coefs_grtn['b2'].n)
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_calc_nrtn() -> _f_qp_theta2',
                          col_x, col_y))
        self.models.update_eonr(self)

        popt, pcov = self._curve_fit_opt(self.models.qp_theta2, x, y,
                                         p0=guess, maxfev=1000, info=info)
        res = y - self.models.qp_theta2(x, *popt)
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
        try:
            pl_l = self.df_ci_temp['pl_l'].item()
            pl_u = self.df_ci_temp['pl_u'].item()
            wald_l = self.df_ci_temp['wald_l'].item()
            wald_u = self.df_ci_temp['wald_u'].item()
            if self.bootstrap_ci is True:
                boot_l = self.df_ci_temp['boot_l'].item()
                boot_u = self.df_ci_temp['boot_u'].item()
        except TypeError as err:
            print(err)
        print('{0} optimum N rate ({1}): {2:.1f} {3} [{4:.1f}, '
              '{5:.1f}] ({6:.1f}% confidence)'
              ''.format(self.onr_name, self.onr_acr, self.eonr, self.unit_nrate, pl_l,
                        pl_u, self.ci_level*100))
        print('Maximum return to N (MRTN): {0}{1:.2f} per {2}'
              ''.format(self.unit_currency, self.mrtn, self.unit_area))

        if self.print_out is True:
            print('Profile likelihood confidence bounds (90%): [{0:.1f}, '
                  '{1:.1f}]'.format(pl_l, pl_u))
            print('Wald confidence bounds (90%): [{0:.1f}, {1:.1f}]'
                  ''.format(wald_l, wald_u))
            print('Bootstrapped confidence bounds (90%): [{0:.1f}, {1:.1f}]\n'
                  ''.format(boot_l, boot_u))

    def _quad_plateau(self, col_x, col_y, rerun=False):
        '''
        Computes quadratic plateau coeficients using numpy.polyfit()

        <col_x (``str``): df column name for x axis
        <col_y (``str``): df column name for y axis
        '''
        df_data = self.df_data.copy()
        x = df_data[col_x].values
        y = df_data[col_y].values
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_quad_plateau() -> _f_quad_plateau', col_x, col_y))
        guess = self._get_guess_qp(rerun=rerun)
        self.models.update_eonr(self)
        #  TODO: Add a try/except to catch a bad guess.. or at least warn the
        # user that the guess is *extremely* sensitive
        popt, pcov = self._curve_fit_opt(self.models.quad_plateau, x, y,
                                         p0=guess, info=info)
        if popt is None or np.any(popt == np.inf) or np.any(pcov == np.inf):
            b0 = unc.ufloat(popt[0], 0)
            b1 = unc.ufloat(popt[1], 0)
            b2 = unc.ufloat(popt[2], 0)
        else:
            b0, b1, b2 = unc.correlated_values(popt, pcov)

        crit_x = -b1.n/(2*b2.n)
        max_y = self.models.quad_plateau(crit_x, b0.n, b1.n, b2.n)
        r2, r2_adj, ss_res, ss_tot, rmse = self._get_rsq(
                self.models.quad_plateau, x, y, popt)
        aic = self._calc_aic(x, y, dist='gamma')

        if rerun is False:
            self.coefs_grtn = {
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
                    'theta2_error': None,
                    'theta2_social': None,
                    'popt_social': None,
                    'ss_res_social': None
                    }
        else:
            self.coefs_grtn_primary = self.coefs_grtn.copy()
            self.coefs_grtn = {}
            self.coefs_grtn = {
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
        y_fert_n = x1 * self.cost_n_fert
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
        self.models.update_eonr(self)
        f_eonr1 = np.poly1d([self.coefs_nrtn['b2'].n,
                            self.coefs_nrtn['theta2'].n,
                            self.coefs_nrtn['b0'].n])
        f_eonr2 = np.poly1d([self.coefs_grtn['b2'].n,
                            self.coefs_grtn['b1'].n,
                            self.coefs_grtn['b0'].n])
        if self.cost_n_social > 0:
            if self.coefs_social['lin_r2'] > self.coefs_social['exp_r2']:
                # subtract only cost of fertilizer
                first_order = self.cost_n_fert + self.coefs_social['lin_mx']
                f_eonr1 = self._modify_poly1d(f_eonr1, 1,
                                              f_eonr1.coef[1] - first_order)
                f_eonr1 = self._modify_poly1d(f_eonr1, 0,
                                              self.coefs_social['lin_b'])
                result = minimize_scalar(-f_eonr1)
                self.f_eonr = f_eonr1
            else:  # add together the cost of fertilizer and cost of social N
                x_max = self.df_data[self.col_n_app].max()
                result = minimize_scalar(self.models.combine_rtn_cost,
                                         bounds=[-100, x_max+100],
                                         method='bounded')
        else:
            first_order = self.coefs_grtn['b1'].n - self.cost_n_fert
            f_eonr2 = self._modify_poly1d(f_eonr2, 1, first_order)
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
        guess = (self.coefs_grtn['b0'].n,
                 self.eonr,
                 self.coefs_grtn['b2'].n)
        self.models.update_eonr(self)
        popt, pcov = self._curve_fit_opt(self.models.qp_theta2, x, y, p0=guess,
                                         maxfev=800)
        self.coefs_nrtn['theta2_error'] = popt[1] - self.eonr

    #  Following are functions used in calculating confidence intervals
    def _bs_statfunction(self, x, y):
        '''
        '''
        maxfev = 1000
        b0 = self.coefs_grtn['b0'].n
        b2 = self.coefs_grtn['b2'].n
        guess = (b0, self.eonr, b2)
        self.models.update_eonr(self)
#        y = self.models.quad_plateau(x, a, b, c) + res
#        try_n = 0
        popt = [None, None, None]
        try:
            popt, _ = self._curve_fit_bs(self.models.qp_theta2, x, y,
                                         p0=guess, maxfev=maxfev)
        except RuntimeError as err:
            print(err)
            maxfev = 10000
            print('Increasing the maximum number of calls to the function to '
                  '{0} before giving up.\n'.format(maxfev))
        if popt[1] is None:
            try:
                popt, _ = self._curve_fit_bs(self.models.qp_theta2, x,
                                             y, p0=guess, maxfev=maxfev)
            except RuntimeError as err:
                print(err)
        return popt[1]

    def _build_df_ci(self):
        '''
        Builds a template to store confidence intervals in a dataframe
        '''
        df_ci = pd.DataFrame(data=[[self.df_data.iloc[0]['location'],
                                    self.df_data.iloc[0]['year'],
                                    self.df_data.iloc[0]['time_n'],
                                    self.price_grain,
                                    self.cost_n_fert,
                                    self.cost_n_social,
                                    self.price_ratio,
                                    0, 0, 0, self.eonr,
                                    self.eonr, self.eonr, self.eonr,
                                    'N/A', 'N/A']],
                             columns=['location', 'year', 'time_n',
                                      'price_grain',
                                      'cost_n_fert', 'cost_n_social',
                                      'price_ratio', 'f_stat', 't_stat',
                                      'level', 'wald_l', 'wald_u',
                                      'pl_l', 'pl_u',
                                      'opt_method_l', 'opt_method_u'])
        return df_ci

    def _calc_sse_full(self, x, y):
        '''
        Calculates the sum of squares across the full set of parameters,
        solving for theta2
        '''
        guess = (self.coefs_grtn['b0'].n,
                 self.eonr,
                 self.coefs_grtn['b2'].n)
        col_x = None  # Perhaps we should keep in col_x/ytil curve_fit runs..?
        col_y = None
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_calc_sse_full() -> _f_qp_theta2',
                          col_x, col_y))
        self.models.update_eonr(self)
        popt, pcov = self._curve_fit_opt(self.models.qp_theta2, x, y, p0=guess,
                                         info=info)
        res = y - self.models.qp_theta2(x, *popt)
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

    def _compute_bootstrap(self, alpha=0.1, n_samples=9999):
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
                                   alpha=alpha, n_samples=n_samples,
                                   method='bca')
        except TypeError:
            if not isinstance(alpha, list):
                print('Unable to compute bootstrap confidence intervals at '
                      'alpha = {0}'.format(alpha))
        return boot_ci

    def _compute_cis(self, col_x, col_y, bootstrap_ci=True):
        '''
        Computes Wald, Profile Likelihood, and Bootstrap confidence intervals.
        '''
        alpha_list = self.alpha_list
        df_ci = self._build_df_ci()
        cols = df_ci.columns
        for alpha in alpha_list:
            pl_l, pl_u, wald_l, wald_u, opt_method_l, opt_method_u =\
                    self._get_likelihood(alpha, col_x, col_y, stat='t')
            level = 1 - alpha
            f_stat = stats.f.ppf(1-alpha, dfn=1, dfd=len(self.df_data)-3)
            t_stat = stats.t.ppf(1-alpha/2, len(self.df_data)-3)
            df_row = pd.DataFrame([[self.df_data.iloc[0]['location'],
                                    self.df_data.iloc[0]['year'],
                                    self.df_data.iloc[0]['time_n'],
                                    self.price_grain,
                                    self.cost_n_fert,
                                    self.cost_n_social,
                                    self.price_ratio,
                                    f_stat, t_stat, level,
                                    wald_l, wald_u,
                                    pl_l, pl_u,
                                    opt_method_l, opt_method_u]],
                                  columns=cols)
            df_ci = df_ci.append(df_row, ignore_index=True)
            if df_row['level'].item() == self.ci_level:
                self.df_ci_temp = df_row
        if bootstrap_ci is True:
            df_ci = self._run_bootstrap(df_ci, alpha_list, n_samples=9999)
        if self.df_ci is None:
            df_ci.insert(loc=0, column='run_n', value=1)
            self.df_ci = df_ci
        else:
            last_run_n = self.df_ci.iloc[-1, :]['run_n']
            df_ci.insert(loc=0, column='run_n', value=last_run_n+1)
            self.df_ci = self.df_ci.append(df_ci, ignore_index=True)
        last_run_n = self.df_ci.iloc[-1, :]['run_n']
        self.df_ci_temp = self.df_ci[(self.df_ci['run_n'] == last_run_n) &
                                     (self.df_ci['level'] == self.ci_level)]

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
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_compute_residuals() -> _f_quad_plateau',
                          col_x, col_y))
        self.models.update_eonr(self)
        if self.base_zero is False:
            popt, pcov = self._curve_fit_opt(self.models.quad_plateau, x, y,
                                             p0=(600, 3, -0.01), info=info)
        else:
            popt, pcov = self._curve_fit_opt(self.models.quad_plateau, x, y,
                                             p0=(0, 3, -0.01), info=info)
        res = y - self.models.quad_plateau(x, *popt)
        res -= res.mean()
        df_temp = pd.DataFrame(data=res, index=df_data.index,
                               columns=['grtn_res'])
        self.df_temp = df_temp
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
#        # if rerun is True, use eonr.coefs_grtn['coef_b']
#        elif rerun is True:
#            guess = (0, round(self.coefs_grtn['coef_b'].n, 0),
#                     round(self.coefs_grtn['coef_c'].n, 0))
        elif rerun is True and self.metric is False:
            guess = (0, 3, -0.01)
        elif rerun is True and self.metric is True:
            guess = (0, 10, -0.03)
        else:
            raise ValueError('<rerun> and eonr.metric should be set to '
                             'True/False.')
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
        # First, get variables stored in the EONR class
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
        while li.tau < crit_stat:
            popt, pcov = self._curve_fit_runtime(
                    lambda x, b0, b2: self.models.qp_theta2(
                            x, b0, li.theta2, b2), x, y,
                    guess=(1, 1), maxfev=800, info=li.info)
            if popt is not None:
                popt = np.insert(popt, 1, li.theta2)
                res = y - self.models.qp_theta2(x, *popt)
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
        # Can't stop when within x of epsilon because cometimes convergence isn't reached
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
        if side == 'lower':
            initial_guess = theta2_opt - pl_guess
        elif side == 'upper':
            initial_guess = theta2_opt + pl_guess
        result = minimize(f, initial_guess, method=method)
#            print(result)
        if pl_guess > 800:
            pl_out = None
        elif result.success is not True:
            return self._run_minimize_pl(f, theta2_opt,
                                         pl_guess*1.05,
                                         method=method,
                                         side=side,
                                         pl_guess_init=pl_guess_init)
        elif result.success is True and side == 'lower':
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
        else:  # finally, return result
            pl_out = result.x[0]
        return pl_out

    def _get_likelihood(self, alpha, col_x, col_y, stat='t',
                        last_ci=[None, None]):
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
        self.models.update_eonr(self)
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
            popt, pcov = self._curve_fit_runtime(
                    lambda x, b0, b2: self.models.qp_theta2(
                            x, b0, theta2, b2), x, y,
                    guess=(1, 1), maxfev=800, info=info)
            if popt is not None:
                popt = np.insert(popt, 1, theta2)
                res = y - self.models.qp_theta2(x, *popt)
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

#        popt, pcov = self._curve_fit_opt(self._f_qp_theta2, x, y, p0=guess, maxfev=800, info=info)
        wald_l, wald_u = self._compute_wald(n, p, alpha)
        pl_guess = (wald_u - self.eonr)  # Adjust +/- init guess based on Wald
        theta2_bias = self.coefs_nrtn['theta2_error']
        theta2_opt = self.eonr + theta2_bias  # check if this should add the 2

        # Lower CI: uses the Nelder-Mead algorithm
        method='Nelder-Mead'
        pl_l = self._run_minimize_pl(_f_like_opt, theta2_opt, pl_guess,
                                     method=method, side='lower')
        pl_u = self._run_minimize_pl(_f_like_opt, theta2_opt, pl_guess,
                                     method=method, side='upper')

        if pl_l is not None:
            pl_l += theta2_bias
        if pl_u is not None:
            pl_u += theta2_bias
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
            df_row = pd.DataFrame([[self.df_data.iloc[0]['location'],
                                    self.df_data.iloc[0]['year'],
                                    self.df_data.iloc[0]['time_n'],
                                    self.price_grain,
                                    self.cost_n_fert,
                                    self.cost_n_social,
                                    self.price_ratio,
                                    f_stat, t_stat, level,
                                    wald_l, wald_u, np.nan, np.nan,
                                    'N/A', 'N/A']],
                                  columns=df_ci.columns)
            df_ci = df_ci.append(df_row, ignore_index=True)
        boot_ci = [None] * ((len(self.ci_list) * 2))
        boot_ci = [self.eonr, self.eonr] + list(boot_ci)
        df_boot = self._parse_boot_ci(boot_ci)
        df_ci = pd.concat([df_ci, df_boot], axis=1)
        if self.df_ci is None:
            df_ci.insert(loc=0, column='run_n', value=1)
            self.df_ci = df_ci
        else:
            last_run_n = self.df_ci.iloc[-1, :]['run_n']
            df_ci.insert(loc=0, column='run_n', value=last_run_n+1)
            self.df_ci = self.df_ci.append(df_ci, ignore_index=True)
        last_run_n = self.df_ci.iloc[-1, :]['run_n']
        self.df_ci_temp = self.df_ci[(self.df_ci['run_n'] == last_run_n) &
                                     (self.df_ci['level'] == self.ci_level)]

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

    def _parse_alpha_list(self, alpha_list):
        '''
        Creates a lower and upper percentile from a list of alpha values. The
        lower is (alpha / 2) and upper is (1 - (alpha / 2)). Required for
        scikits-bootstrap
        '''
        alpha_list_pctle = []
        for alpha in alpha_list:
            pctle_l = alpha / 2
            pctle_u = 1 - (alpha / 2)
            alpha_list_pctle.extend([pctle_l, pctle_u])
        return alpha_list_pctle

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
        df_boot = pd.DataFrame({'boot_l': boot_l,
                                'boot_u': boot_u})
        return df_boot

    def _run_bootstrap(self, df_ci, alpha_list, n_samples=9999):
        '''
        Calls the _compute_bootstrap() function.
        '''
        pctle_list = self._parse_alpha_list(alpha_list)
        boot_ci = self._compute_bootstrap(alpha=pctle_list,
                                          n_samples=n_samples)
        if boot_ci is None:
            boot_ci = []
            for pctle in pctle_list:
                boot_ci_temp = self._compute_bootstrap(alpha=pctle,
                                                       n_samples=n_samples)
                boot_ci.append(boot_ci_temp)
        if boot_ci is None:
            boot_ci = [None] * ((len(self.ci_list) * 2) + 2)
        boot_ci = [self.eonr, self.eonr] + list(boot_ci)
        df_boot = self._parse_boot_ci(boot_ci)
        df_ci = pd.concat([df_ci, df_boot], axis=1)
        return df_ci

    def calc_delta(self, df_results=None):
        '''Calculates the change in EONR among economic scenarios

        Parameters:
            df_results (``Pandas dataframe``, optional): The dataframe
                containing the results from ``EONR.calculate_eonr()``
                (default: None).

        Returns:
            df_out "The dataframe with the newly inserted EONR delta."

        Note:
            ``EONR.calc_delta()`` filters all data by location, year, and
            nitrogen timing, then the "delta" is calculated as the difference
            relative to the economic scenario resulting in the highest EONR.

        '''
        if df_results is None:
            df = self.df_results.unique()
        else:
            df = df_results.copy()

        years = df['year'].unique()
        years.sort()
        df_out = None
        for year in years:
            df_year = df[df['year'] == year]
            locs = df_year['location'].unique()
            locs.sort()
            for loc in locs:
                df_loc = df_year[df_year['location'] == loc]
                times = df_loc['time_n'].unique()
                for time in times:
                    df_yloct = df_loc[df_loc['time_n'] == time]
                    eonr_base = df_yloct['eonr'].max()  # lowest fert:grain rat
                    eonr_delta = df_yloct['eonr'] - eonr_base
                    df_yloct.insert(8, 'eonr_delta', eonr_delta)
                    if df_out is None:
                        df_out = pd.DataFrame(columns=df_yloct.columns)
                    df_out = df_out.append(df_yloct)
        return df_out

    def calculate_eonr(self, df, col_n_app=None, col_yld=None,
                       col_crop_nup=None, col_n_avail=None,
                       col_year=None, col_location=None, col_time_n=None,
                       bootstrap_ci=True):
        '''Calculates the EONR and its confidence intervals

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
                intervals take the most time to compute (default: True).

        Note:
            ``col_n_app`` and ``col_yld`` are required by ``EONR``, but not
            necessarily by ``EONR.calculate_eonr()``. They must either be set
            during the initialization of EONR(), or be passed in this method.

        Note:
            ``col_crop_nup`` and ``col_n_avail`` are required to calculate the
            socially optimum nitrogen rate, SONR. The SONR is the optimum
            nitrogen rate considering the social cost of nitrogen, so
            therefore, ``EONR.cost_n_social`` must also be set.

        Note:
            ``col_year``, ``col_location``, and ``col_time_n`` are purely
            optional. They only affect the titles and axes labels of the plots.

        '''
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
        self._compute_R(col_x=self.col_n_app, col_y='grtn')
        self._theta2_error()
        if self.eonr > self.df_data[self.col_n_app].max():
            print('\n{0} is past the point of available data, so confidence '
                  'bounds are not being computed'.format(self.onr_acr))
            self._handle_no_ci()
        else:
            self._compute_residuals()
            self._compute_cis(col_x=self.col_n_app, col_y='grtn',
                              bootstrap_ci=bootstrap_ci)
        self._build_mrtn_lines()
        if self.print_out is True:
            self._print_grtn()
        self._print_results()
        if self.base_zero is True:
            base_zero = self.coefs_grtn_primary['b0'].n
        else:
            base_zero = self.coefs_grtn['b0'].n
        results = [[self.price_grain, self.cost_n_fert, self.cost_n_social,
                    self.price_ratio, self.location, self.year, self.time_n,
                    base_zero, self.eonr,
                    self.coefs_nrtn['theta2_error'],
                    self.ci_level, self.df_ci_temp['wald_l'].item(),
                    self.df_ci_temp['wald_u'].item(),
                    self.df_ci_temp['pl_l'].item(),
                    self.df_ci_temp['pl_u'].item(),
                    self.df_ci_temp['boot_l'].item(),
                    self.df_ci_temp['boot_u'].item(),
                    self.mrtn, self.coefs_grtn['r2_adj'],
                    self.coefs_grtn['rmse'],
                    self.coefs_grtn['max_y'],
                    self.coefs_grtn['crit_x'],
                    self.results_temp['grtn_y_int'],
                    self.results_temp['scn_lin_r2'],
                    self.results_temp['scn_lin_rmse'],
                    self.results_temp['scn_exp_r2'],
                    self.results_temp['scn_exp_rmse']]]

        self.df_results = self.df_results.append(pd.DataFrame(
                results, columns=self.df_results.columns),
                ignore_index=True)

    def plot_eonr(self, ci_type='profile-likelihood', ci_level=None,
                  run_n=None, x_min=None, x_max=None, y_min=None, y_max=None,
                  style='ggplot'):
        '''Plots EONR, MRTN, GRTN, net return, and nitrogen cost

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
            style (``str``, optional): The style of the plolt; can be any of
                the options supported by ``matplotlib``

        Note:
            ``x_min``, ``x_max``, ``y_min``, and ``y_max`` are set by
            Matplotlib if left as ``None``.

        .. _matplotlib:
            https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html

        '''
        if self.plotting_tools is None:
            self.plotting_tools = Plotting_tools(self)
        else:
            self.plotting_tools.update_eonr(self)
        self.plotting_tools.plot_eonr(ci_type=ci_type, ci_level=ci_level,
                                      run_n=run_n, x_min=x_min, x_max=x_max,
                                      y_min=y_min, y_max=y_max, style=style)
        self.fig_eonr = self.plotting_tools.fig_eonr

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

        '''
        self.plotting_tools.plot_save(fname=fname, base_dir=base_dir, fig=fig,
                                      dpi=dpi)

    def plot_tau(self, y_axis='t_stat', emphasis='profile-likelihood',
                 run_n=None):
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

        '''
        if self.plotting_tools is None:
            self.plotting_tools = Plotting_tools(self)
        else:
            self.plotting_tools.update_eonr(self)
        self.plotting_tools.plot_tau(y_axis=y_axis, emphasis=emphasis,
                                     run_n=run_n)
        self.fig_tau = self.plotting_tools.fig_tau

    def print_results(self):
        '''Prints the results of the optimum nitrogen rate computation

        '''
        self._print_results()

    def set_column_names(self, col_n_app=None, col_yld=None, col_crop_nup=None,
                         col_n_avail=None, col_year=None,
                         col_location=None, col_time_n=None):
        '''Sets the column name(s) for ``EONR.df_data``

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

        Note:
            Year, location, or nitrogen timing (used for titles and axes labels
            for plotting).

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
        self._find_trial_details()  # Use new col_name(s) to update details

    def set_trial_details(self, year=None, location=None, n_timing=None):
        '''Sets the year, location, or nitrogen timing

        Parameters:
            year (``str`` or ``int``, optional): Year of experimental trial
                (default: None)
            location (``str`` or ``int``, optional): Location of experimental
                trial (default: None)
            n_timing (``str`` or ``int``, optional): Nitrogen timing of
                experimental trial (default: None)

        Note:
            Year, location, or nitrogen timing (used for titles and axes labels
            for plotting).

        '''
        if year is not None:
            self.year = int(year)
        if location is not None:
            self.location = location
        if n_timing is not None:
            self.n_timing = n_timing

    def update_econ(self, cost_n_fert=None, cost_n_social=None,
                    price_grain=None):
        '''Sets or resets the nitrogen costs or grain price

        Parameters:
            cost_n_fert (``float``, optional): Cost of nitrogen fertilizer
                (default: None).
            cost_n_social (``float``, optional): Cost of pollution caused by
                excess nitrogen (default: None).
            price_grain (``float``, optional): Price of grain (default: None).

        Note:
            ``update_econ()`` recomputes the price ratio based on the passed
            information, then adjusts/renames the lowest level folder in the
            base directory, ``EONR.base_dir``, based on to the ratio. The
            folder name is set according to the economic scenario (useful when
            running ``EONR`` for many different economic scenarios then
            plotting and saving results for each scenario). See examples below.

            Example 1 (*how folder name is set when social cost of nitrogen is
            zero*): folder name will be set to **"trad_0010"** if
            ``cost_n_social == 0`` and ``price_ratio == 0.10``, coresponding to
            *"trad"* and *"0010"* in the folder name, respectively.

            Example 2 (*how folder name is set when social cost of nitrogen is
            greater than zero*): folder name will be set to
            **"social_154_1100"** if ``cost_n_social > 0``,
            ``price_ratio = 15.4``, and ``cost_n_social = 1.10``, corresponding
            to *"social"*, *"154"*, and *"1100"* in the folder name,
            respectively.

        '''
        if cost_n_fert is not None:
            self.cost_n_fert = cost_n_fert  # in USD per lb
        if cost_n_social is not None:
            self.cost_n_social = cost_n_social  # in USD per lb lost
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

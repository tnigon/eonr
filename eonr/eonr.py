# -*- coding: utf-8 -*-
"""
© 2019 Regents of the University of Minnesota. All rights reserved.

EONR is copyrighted by the Regents of the University of Minnesota. It can
be freely used for educational and research purposes by non-profit
institutions and US government agencies only. Other organizations are allowed
to use EONR only for evaluation purposes, and any further uses will require
prior approval. The software may not be sold or redistributed without prior
approval. One may make copies of the software for their use provided that the
copies, are not sold or distributed, are used under the same terms and
conditions.
As unestablished research software, this code is provided on an "as is" basis
without warranty of any kind, either expressed or implied. The downloading, or
executing any part of this software constitutes an implicit agreement to these
terms. These terms and conditions are subject to change at any time without
prior notice.
"""

import matplotlib.colors as colors
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
import seaborn as sns
import uncertainties as unc
from uncertainties import unumpy as unp
import warnings

plt.style.use('ggplot')


class EONR(object):
    '''
    Calculates Economic Optimum N Rate given table of N applied and yield

    <cost_n_fert> --> Cost of N fertilizer
    <cost_n_social> --> Social cost of N fertilier
    <price_grain> --> Price of grain
    <col_n_app> --> Column name pointing to the rate of applied N fertilizer
        data
    <col_yld> --> Column name pointing to the grain yield data
    <col_crop_nup> --> Column name pointing to crop N uptake data
    <col_n_soil_fert> --> Column name pointing to available soil N plus
        fertilizer
    <unit_currency> --> String describing the curency unit (default: '$')
    <unit_fert> --> String describing the "fertilizer" unit (default: 'lbs')
    <unit_grain> --> String describing the "grain" unit (default: 'bu')
    <unit_area> --> String descibing the "area" unit (default: 'ac')
    <model> --> Statistical model used to fit N rate response. 'quad_platau' =
        quadratic plateau; 'lin_plateau = linear plateau (default:
        'quad_plateau')
    <ci_level> --> Confidence interval level to save in eonr.df_results and to
        display in the EONR plot (confidence intervals are calculated at many
        alpha levels)
    <base_dir> --> Base file directory when saving results
    <base_zero> --> Determines if gross return to N is expressed as an absolute
        value or relative to the yield/return at the zero rate (i.e., at the
        y-intercept of the absolute gross response to N curve); default: False
    <print_out> --> Determines if "non-essential" results are printed in the
        Python console (default: False)
    '''
    def __init__(self,
                 cost_n_fert=0.5,
                 cost_n_social=0,
                 price_grain=3.00,
                 col_n_app='rate_n_applied_lbac',
                 col_yld='yld_grain_dry_buac',
                 col_crop_nup='nup_total_lbac',
                 col_nup_soil_fert='soil_plus_fert_n_lbac', unit_currency='$',
                 unit_fert='lbs', unit_grain='bu', unit_area='ac',
                 model='quad_plateau', ci_level=0.9, base_dir=None,
                 base_zero=False, print_out=False):
        self.df_data = None
        self.cost_n_fert = cost_n_fert
        self.cost_n_social = cost_n_social
        self.price_grain = price_grain
        self.price_ratio = round((cost_n_fert+cost_n_social) / price_grain, 3)
        self.col_n_app = col_n_app
        self.col_yld = col_yld
        self.col_crop_nup = col_crop_nup
        self.col_nup_soil_fert = col_nup_soil_fert
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
        self.coefs_grtn_lp = {}
        self.coefs_grtn_qp = {}
        self.coefs_grtn = {}
        self.coefs_grtn_primary = {}  # only used if self.base_zero is True
        self.coefs_nrtn = {}
        self.coefs_social = {}
        self.coefs_n_uptake = {}
        self.results_temp = {}

        self.mrtn = None
        self.eonr = None
        self.df_ci = None

        self.fig_eonr = None
        self.fig_tau = None
        self.palette = self._seaborn_palette(color='muted', cmap_len=10)
        self.linspace_cost_n_fert = None
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

    def _set_df(self, df):
        '''
        Basic setup for dataframe
        '''
        self.df_data = df.copy()
        self.location = df.iloc[0].location
        self.year = df.iloc[0].year
        self.time_n = df.iloc[0].time_n
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

    #  Following are the algebraic functions
    def _f_exp(self, x, a, b, c):
        '''Exponential 3-param function.'''
        return a * np.exp(b * x) + c

    def _f_poly(self, x, c, b, a=None):
        '''
        Polynomial function (up to 3 parameters)
        '''
        if a is None:
            a = 0
        return a*x**2 + b*x + c

    def _f_quad_plateau(self, x, b0, b1, b2):
        '''
        Quadratic plateau function

        Y =     b0 + b1*x +b2*(x^2) + error
                                            --> if x < <crit_x>
                b0 - (b1^2)/(4*b2) + error
                                            --> if x >= <crit_x>
        where crit_x = -b1/(2*b2)
        '''
        crit_x = -b1/(2*b2)
        if isinstance(x, int) or isinstance(x, float):
            y = 0
            y += (b0 + b1*x + b2*(x**2)) * (x < crit_x)
            y += (b0 - (b1**2) / (4*b2)) * (x >= crit_x)
            return y
        else:
            array_temp = np.zeros(len(x))
            array_temp += (b0 + b1*x + b2*(x**2)) * (x < crit_x)
            array_temp += (b0 - (b1**2) / (4*b2)) * (x >= crit_x)
            return array_temp

    def _f_qp_theta2(self, x, b0, theta2, b2):
        '''
        Quadratic plateau function using theta2 (EOR) as one of the parameters

        theta2 = (R-b1)/(2*b2) = EOR
            where R is the price ratio and b2 = beta2
        rearrange to get:

        b1_hat = -(2*theta2*b2-R)

        b1_hat can replace b1 in the original quadratic plateau model to
        incorporate theta2 (EOR) as one of the parameters. This is desireable
        because we can use scipy.optimize.curve_fit() to compute the covariance
        matrix and estimate confidence intervals for theta2 directly.
        '''
        R = self.R
        # We coult calculate crit_x and we should get the best fit..
#        b1 = -(2*theta2*b2 - R)
#        crit_x = (-b1/(2*(b2)))
        # But we will always know crit_x from fitting the gross return model
        crit_x = self.coefs_grtn['crit_x']  # and we know it's accurate
        if isinstance(x, int) or isinstance(x, float):
            y = 0
            y += (b0 - ((2*theta2*b2) - R)*x + b2*(x**2)) * (x < crit_x)
            y += (b0 - ((2*theta2*b2 - R)**2) / (4*b2)) * (x >= crit_x)
            return y
        else:
            array_temp = np.zeros(len(x))
            array_temp += (b0 - ((2*theta2*b2) - R)*x + b2*(x**2)) * (x < crit_x)
            array_temp += (b0 - ((2*theta2*b2 - R)**2) / (4*b2)) * (x >= crit_x)
            return array_temp

    def _f_combine_rtn_cost(self, x):
        '''
        Social cost of N can take on an exponential form, while fertilizer
        cost is a first-order linear function. This function combines
        linear and exponential (non-linear) functions, resulting in a
        single, non-linear function definition

        A linear model can be described as:
            y = a**x + b*x + c
        An exponential (nonlinear) model can be described as:
            y = a * e^b*x + c
        This function combines the two to get terms for a single
        exponential model
        Returns x, the result of combining the two
        '''
        # Additions
        b0 = self.coefs_grtn['b0'].n
        b1 = self.coefs_grtn['b1'].n
        b2 = self.coefs_grtn['b2'].n
        gross_rtn = self._f_quad_plateau(x, b0=b0, b1=b1, b2=b2)
        # Subtractions
        fert_cost = x * self.cost_n_fert
        if self.coefs_social['lin_r2'] > self.coefs_social['exp_r2']:
            lin_b = self.coefs_social['lin_b']
            lin_mx = self.coefs_social['lin_mx']
            social_cost = self._f_poly(x, c=lin_b, b=lin_mx)
        else:
            exp_a = self.coefs_social['exp_a']
            exp_b = self.coefs_social['exp_b']
            exp_c = self.coefs_social['exp_c']
            social_cost = self._f_exp(x, a=exp_a, b=exp_b, c=exp_c)
        result = gross_rtn - fert_cost - social_cost
        return -result

    def _f_qp_gross_theta2(self, x, b0, theta2, b2):
        '''
        Quadratic plateau function using theta2 (EOR) as one of the parameters
        and not considering any additonal economic constants (e.g.,
        grain:fertilizer price ratio).

        Note:
            theta2 = -b1/(2*b2)
        can be rearranged to get theta2 as one of the parameters of the model:
            b1 = -(2*theta2*b2)*x
        and
            -b1**2 / (4*b2)
        becomes
            -(theta2**2*b2))
        '''
#        R = self.R
#        b1 = -(2*theta2*b2 - R)
        if isinstance(x, int) or isinstance(x, float):
            y = 0
            y += (b0 + (2*b2*theta2)*x + b2*(x**2)) * (x < theta2)
            y += (b0 - (theta2**2*b2)) * (x >= theta2)
            return y
        else:
            array_temp = np.zeros(len(x))
            array_temp += (b0 - (2*b2*theta2)*x + b2*(x**2)) * (x < theta2)
            array_temp += (b0 - (theta2**2*b2)) * (x >= theta2)
            return array_temp

    def _f_qp_net(self, x, b0, theta2, b2):
        '''
        Quadratic plateau function to compute Net response to N

        This works for generating a curve for x values given all parameters,
        but will not work for curve-fitting unless the <R> component is
        subtracted from all gross return ('grtn') data for its respective
        x/N rate

        Use:
            b0 = my_eonr.coefs_grtn['b0'].n
            theta2 = my_eonr.eonr
            b2 = my_eonr.coefs_grtn['b2'].n
            y1 = _f_qp_net(x, b0, theta2, b2)
            sns.scatterplot(x, y1)
        '''
        R = self.R
        b1 = -(2*theta2*b2 - R)
        crit_x = (-b1/(2*(b2)))
        array_temp = np.zeros(len(x))
        array_temp += (b0 + -(2*theta2*b2)*x + b2*(x**2)) * (x < crit_x)
        array_temp += (b0 - ((crit_x**2)*b2) - (R*x)) * (x >= crit_x)
        return array_temp

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
            popt, pcov = self._curve_fit_opt(self._f_exp, x, y,
                                             p0=(guess_a, guess_b, guess_c),
                                             maxfev=1500, info=info)
        except RuntimeError as err:
            print('\n{0}\nTrying a new guess before giving up..'.format(err))
            pass

        if popt is None:
            try:
                popt, pcov = self._curve_fit_opt(self._f_exp, x, y,
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
            exp_r2, _, _, _, exp_rmse = self._get_rsq(self._f_exp, x, y,
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

            guess = (self.coefs_grtn['b0'].n,
                     self.coefs_grtn['crit_x'],
                     self.coefs_grtn['b2'].n)

            popt, pcov = self._curve_fit_runtime(self._f_qp_theta2, x,
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
                popt, pcov = self._curve_fit_runtime(self._f_qp_theta2,
                                                     x, y, guess, maxfev=800)
                dif = abs(popt[1] - self.eonr)
            res = y - self._f_qp_theta2(x, *popt)
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
            func = self._f_quad_plateau
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
        col_n_app    --> column heading for N applied (str)
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

        popt, pcov = self._curve_fit_opt(self._f_qp_theta2, x, y,
                                         p0=guess, maxfev=1000, info=info)
        res = y - self._f_qp_theta2(x, *popt)
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
        self.df_data['resid_n'] = (self.df_data[self.col_nup_soil_fert] -
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

        <col_x> --> df column name for x axis
        <col_y> --> df column name for y axis
        '''
        df_data = self.df_data.copy()
        x = df_data[col_x].values
        y = df_data[col_y].values
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_quad_plateau() -> _f_quad_plateau', col_x, col_y))
        guess = self._get_guess_qp(rerun=rerun)
        #  TODO: Add a try/except to catch a bad guess.. or at least warn the
        # user that the guess is *extremely* sensitive
        popt, pcov = self._curve_fit_opt(self._f_quad_plateau, x, y,
                                         p0=guess, info=info)
        if popt is None or np.any(popt == np.inf) or np.any(pcov == np.inf):
            b0 = unc.ufloat(popt[0], 0)
            b1 = unc.ufloat(popt[1], 0)
            b2 = unc.ufloat(popt[2], 0)
        else:
            b0, b1, b2 = unc.correlated_values(popt, pcov)

        crit_x = -b1.n/(2*b2.n)
        max_y = self._f_quad_plateau(crit_x, b0.n, b1.n, b2.n)
        r2, r2_adj, ss_res, ss_tot, rmse = self._get_rsq(
                self._f_quad_plateau, x, y, popt)
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
                result = minimize_scalar(self._f_combine_rtn_cost,
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
        popt, pcov = self._curve_fit_opt(self._f_qp_theta2, x, y, p0=guess,
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
#        y = self._f_quad_plateau(x, a, b, c) + res
#        try_n = 0
        popt = [None, None, None]
        try:
            popt, _ = self._curve_fit_bs(self._f_qp_theta2, x, y,
                                         p0=guess, maxfev=maxfev)
        except RuntimeError as err:
            print(err)
            maxfev = 10000
            print('Increasing the maximum number of calls to the function to '
                  '{0} before giving up.\n'.format(maxfev))
        if popt[1] is None:
            try:
                popt, _ = self._curve_fit_bs(self._f_qp_theta2, x,
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
        popt, pcov = self._curve_fit_opt(self._f_qp_theta2, x, y, p0=guess,
                                         info=info)
        res = y - self._f_qp_theta2(x, *popt)
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
        if self.base_zero is False:
            popt, pcov = self._curve_fit_opt(self._f_quad_plateau, x, y,
                                             p0=(600, 3, -0.01), info=info)
        else:
            popt, pcov = self._curve_fit_opt(self._f_quad_plateau, x, y,
                                             p0=(0, 3, -0.01), info=info)
        res = y - self._f_quad_plateau(x, *popt)
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

        <n> --> number of samples
        <p> --> number of parameters
        <s2c> --> the variance of the EONR value(notation from Gallant, 1987)

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
            self.col_x = col_x
            self.col_y = col_y
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
                    lambda x, b0, b2: self._f_qp_theta2(
                            x, b0, li.theta2, b2), x, y,
                    guess=(1, 1), maxfev=800, info=li.info)
            if popt is not None:
                popt = np.insert(popt, 1, li.theta2)
                res = y - self._f_qp_theta2(x, *popt)
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
                    lambda x, b0, b2: self._f_qp_theta2(
                            x, b0, theta2, b2), x, y,
                    guess=(1, 1), maxfev=800, info=info)
            if popt is not None:
                popt = np.insert(popt, 1, theta2)
                res = y - self._f_qp_theta2(x, *popt)
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

    #  Following are plotting functions
    def _add_labels(self, g, x_max=None):
        '''
        Adds EONR and economic labels to the plot
        '''
        if x_max is None:
            _, x_max = g.fig.axes[0].get_xlim()
        boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
        el = mpatches.Ellipse((0, 0), 0.3, 0.3, angle=50, alpha=0.5)
        g.ax.add_artist(el)
        label_eonr = '{0}: {1:.0f} {2}\nMRTN: ${3:.2f}'.format(
                self.onr_acr, self.eonr, self.unit_nrate, self.mrtn)
        if self.eonr <= x_max:
            g.ax.plot([self.eonr], [self.mrtn], marker='.', markersize=15,
                      color=self.palette[2], markeredgecolor='white',
                      label=label_eonr)
        if self.eonr > 0 and self.eonr < self.df_data[self.col_n_app].max():
            g.ax.annotate(
                label_eonr,
                xy=(self.eonr, self.mrtn), xytext=(-80, -30),
                textcoords='offset points', ha='left', va='top', fontsize=8,
                color='#333333',
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5)),
                arrowprops=dict(arrowstyle='-|>',
                                color="grey",
                                patchB=el,
                                shrinkB=10,
                                connectionstyle='arc3,rad=-0.3'))
        else:
            g.ax.annotate(
                label_eonr,
                xy=(0, 0), xycoords='axes fraction', xytext=(0.98, 0.05),
                ha='right', va='bottom', fontsize=8, color='#333333',
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5)))

        label_econ = ('Grain price: ${0:.2f}\nN fertilizer cost: ${1:.2f}'
                      ''.format(self.price_grain, self.cost_n_fert))

        if self.cost_n_social > 0 and self.metric is True:
            label_econ = ('{0}\nSocial cost of N: ${1:.2f}\nPrice ratio: '
                          '{2:.2f}'.format(label_econ, self.cost_n_social,
                                           self.price_ratio))
        elif self.cost_n_social > 0 and self.metric is False:
            label_econ = ('{0}\nSocial cost of N: ${1:.2f}\nPrice ratio: '
                          '{2:.3f}'.format(label_econ, self.cost_n_social,
                                           self.price_ratio))
        elif self.cost_n_social == 0 and self.metric is True:
            label_econ = ('{0}\nPrice ratio: {1:.2f}'
                          ''.format(label_econ, self.price_ratio))
        else:
            label_econ = ('{0}\nPrice ratio: {1:.3f}'
                          ''.format(label_econ, self.price_ratio))

        if self.base_zero is True:
            label_econ = ('{0}\nBase zero: {1}{2:.2f}'.format(
                    label_econ, self.unit_currency,
                    self.coefs_grtn_primary['b0'].n))
        g.ax.annotate(
            label_econ,
            xy=(0, 1), xycoords='axes fraction', xytext=(0.98, 0.95),
            horizontalalignment='right', verticalalignment='top',
            fontsize=7, color='#333333',
            bbox=dict(boxstyle=boxstyle_str,
                      fc=(1, 1, 1), ec=(0.5, 0.5, 0.5), alpha=0.75))
        return g

    def _add_title(self, g, size_font=12):
        '''
        Adds the title to the plot
        '''
        title_text = ('{0} {1} - {2} N Fertilizer Timing'
                      ''.format(self.year, self.location, self.time_n))
        divider = make_axes_locatable(g.ax)
        cax = divider.append_axes("top", size="15%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.set_facecolor('#7b7b7b')
        text_obj = AnchoredText(title_text, loc=10, frameon=False,
                                prop=dict(backgroundcolor='#7b7b7b',
                                          size=size_font, color='white',
                                          fontweight='bold'))
        size_font = self._set_title_size(g, title_text, size_font, coef=1.15)
        text_obj.prop.set_size(size_font)
        cax.add_artist(text_obj)
        return g

    def _add_ci(self, g, ci_type='profile-likelihood', ci_level=None):
        '''
        Adds upper and lower confidence curves to the EONR plot
        <ci_type> --> type of confidence interval to display
            'profile-likelihood' (default) is the profile likelihood confidence
            interval (see Cook and Weisberg, 1990)
            'wald' is the Wald (uniform) confidence interval
            if the profile-likelihood confidence interval is available, it is
            recommended to use this for EONR data
        <ci_level> --> the confidence level to display
        '''
        msg = ('Please choose either "profile-likelihood" or "wald" for '
               '<ci_type>')
        ci_types = ['profile-likelihood', 'wald', 'bootstrap']
        assert ci_type.lower() in ci_types, msg
        if ci_type == 'profile-likelihood':
            ci_l = self.df_ci_temp['pl_l'].item()
            ci_u = self.df_ci_temp['pl_u'].item()
        elif ci_type == 'wald':
            ci_l = self.df_ci_temp['wald_l'].item()
            ci_u = self.df_ci_temp['wald_u'].item()
        elif ci_type == 'bootstrap':
            ci_l = self.df_ci_temp['boot_l'].item()
            ci_u = self.df_ci_temp['boot_u'].item()

        g.ax.axvspan(ci_l, ci_u, alpha=0.1, color='#7b7b7b')  # alpha is trans
        if ci_level is None:
            ci_level = self.ci_level
        label_ci = ('Confidence ({0:.2f})'.format(ci_level))

        if self.eonr <= self.df_data[self.col_n_app].max():
            alpha_axvline = 1
            g.ax.axvline(ci_l, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=label_ci, alpha=alpha_axvline)
            g.ax.axvline(ci_u, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=None, alpha=alpha_axvline)
            g.ax.axvline(self.eonr,
                         linestyle='--',
                         linewidth=1.5,
                         color='#555555',
                         label='EONR',
                         alpha=alpha_axvline)
        else:
            alpha_axvline = 0
            g.ax.axvline(0, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=label_ci, alpha=alpha_axvline)
            g.ax.axvline(0, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=None, alpha=alpha_axvline)
            g.ax.axvline(0,
                         linestyle='--',
                         linewidth=1.5,
                         color='#555555',
                         label='EONR',
                         alpha=alpha_axvline)
        return g

    def _cmap_to_palette(self, cmap):
        '''
        Converts matplotlib cmap to seaborn color palette
        '''
        palette = []
        for row in cmap:
            palette.append(tuple(row))
        return palette

    def _color_to_palette(self, color='b', cmap_len=3):
        '''
        Converts generic color into a Seaborn color palette
        <color> can be a string abbreviation for a given color (e.g.,
        'b' = blue);
        <color> can also be a colormap as an array (e.g.,
        plt.get_cmap('viridis', 3))
        '''
        try:
            color_array = colors.to_rgba_array(color)
            palette = self._cmap_to_palette(color_array)
        except ValueError:  # convert to colormap
            color_array = plt.get_cmap(color, cmap_len)
            palette = self._cmap_to_palette(color_array.colors)
        except TypeError:  # convert to palette
            palette = self._cmap_to_palette(colors)
        return palette

    def _draw_lines(self, g, palette):
        '''
        Draws N cost on plot
        '''
        g.ax.plot(self.linspace_qp[0],
                  self.linspace_qp[1],
                  color=palette[0],
                  linestyle='-.',
                  label='Gross return to N')
        g.ax.plot(self.linspace_rtn[0],
                  self.linspace_rtn[1],
                  color=palette[2],
                  linestyle='-',
                  label='Net return to N')
        if self.cost_n_social > 0:
            # Social cost of N is net NUP * SCN (NUP - inorganic N)
            pal2 = self._seaborn_palette(color='hls', cmap_len=10)[1]
            g.ax.plot(self.linspace_cost_n_social[0],
                      self.linspace_cost_n_social[1],
                      color=pal2,
                      linestyle='-',
                      linewidth=1,
                      label='Social cost from res. N')
            total = self.linspace_cost_n_fert[1] +\
                self.linspace_cost_n_social[1]
            g.ax.plot(self.linspace_cost_n_social[0],
                      total,
                      color=palette[3],
                      linestyle='--',
                      label='Total N cost')
        else:
            g.ax.plot(self.linspace_cost_n_fert[0],
                      self.linspace_cost_n_fert[1],
                      color=palette[3],
                      linestyle='--',
                      label='N fertilizer cost')
        return g

    def _modify_axes_labels(self, fontsize=14):
        plt.ylabel('Return to N ({0})'.format(self.unit_rtn),
                   fontweight='bold',
                   fontsize=fontsize)
        xlab_obj = plt.xlabel('N Rate ({0})'.format(self.unit_nrate),
                              fontweight='bold',
                              fontsize=fontsize)
        plt.getp(xlab_obj, 'color')

    def _modify_axes_pos(self, g):
        '''
        Modifies axes positions
        '''
        box = g.ax.get_position()
        g.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return g

    def _modify_minmax_axes(self, g, x_min=None, x_max=None, y_min=None,
                            y_max=None):
        '''
        Attempts to modify the min and max axis values
        <x_min> and <x_max> OR <y_min> and <y_max> must be provided, otherwise
        nothing will be set
        '''
        try:
            x_min
            x_max
            x_true = True
        except NameError:
            x_true = False
        try:
            y_min
            y_max
            y_true = True
        except NameError:
            y_true = False
        if x_true is True:
            g.fig.axes[0].set_xlim(x_min, x_max)
        if y_true is True:
            g.fig.axes[0].set_ylim(y_min, y_max)
        return g

    def _modify_plot_params(self, g, plotsize_x=7, plotsize_y=4, labelsize=11):
        '''
        Modifies plot size, changes font to bold, and sets tick/label size
        '''
        fig = plt.gcf()
        fig.set_size_inches(plotsize_x, plotsize_y)
        plt.rcParams['font.weight'] = 'bold'
        g.ax.xaxis.set_tick_params(labelsize=labelsize)
        g.ax.yaxis.set_tick_params(labelsize=labelsize)
        g.fig.subplots_adjust(top=0.9, bottom=0.135, left=0.11, right=0.99)
        return g

    def _place_legend(self, g, loc='upper left', facecolor='white'):
        h, leg = g.ax.get_legend_handles_labels()
        if self.cost_n_social == 0:
            order = [2, 3, 4, 0]
            handles = [h[idx] for idx in order]
            labels = [leg[idx] for idx in order]
        else:
            order = [2, 3, 4, 5, 0]
            handles = [h[idx] for idx in order]
            labels = [leg[idx] for idx in order]
        patch_ci = mpatches.Patch(facecolor='#7b7b7b', edgecolor='k',
                                  alpha=0.3, fill=True, linewidth=0.5)

        handles[-1] = patch_ci
        leg = g.ax.legend(handles=handles,
                          labels=labels,
                          loc=loc,
                          frameon=True,
                          framealpha=0.75,
                          facecolor='white',
                          fancybox=True,
                          borderpad=0.5,
                          edgecolor=(0.5, 0.5, 0.5),
                          prop={
                                  'weight': 'bold',
                                  'size': 7
                                  })
        for text in leg.get_texts():
            plt.setp(text, color='#333333')
        return g

    def _plot_points(self, col_x, col_y, data, palette, ax=None, s=20,
                     zorder=None):
        '''
        Plots points on figure
        '''
        if ax is None:
            g = sns.scatterplot(x=col_x,
                                y=col_y,
                                color=palette,
                                legend=False,
                                data=data,
                                s=s,
                                zorder=zorder)
        else:
            g = sns.scatterplot(x=col_x,
                                y=col_y,
                                color=palette,
                                legend=False,
                                data=data,
                                ax=ax,
                                s=s,
                                zorder=zorder)
        return g

    def _seaborn_palette(self, color='hls', cmap_len=8):
        '''
        '''
        palette = sns.color_palette(color, cmap_len)
        return palette

    def _set_style(self, style):
        '''
        Make sure style is supported by matplotlib
        '''
        try:
            plt.style.use(style)
        except OSError as err:
            print(err)
            print('Using "ggplot" style instead..'.format(style))
            plt.style.use('ggplot')

    def _pci_modify_axes_labels(self, y_axis, fontsize=11):
        if y_axis == 't_stat':
            y_label = r'$|\tau|$ (T statistic)'
        elif y_axis == 'f_stat':
            y_label = r'$|\tau|$ (F statistic)'
        else:
            y_label = 'Confidence (%)'
        plt.ylabel(y_label,
                   fontweight='bold',
                   fontsize=fontsize)
        xlab_obj = plt.xlabel('True {0} ({1})'.format(self.onr_acr,
                                                      self.unit_nrate),
                              fontweight='bold',
                              fontsize=fontsize)
        plt.getp(xlab_obj, 'color')

    def _pci_place_legend_tau(self, g, loc='lower left', facecolor='white'):
        h, leg = g.ax.get_legend_handles_labels()
        order = [0, 2, 4]
        handles = [h[idx] for idx in order]
        labels = [leg[idx] for idx in order]
        leg = g.ax.legend(handles=handles,
                          labels=labels,
                          loc=loc,
                          frameon=True,
                          framealpha=0.75,
                          facecolor='white',
                          fancybox=True,
                          borderpad=0.5,
                          edgecolor=(0.5, 0.5, 0.5),
                          prop={
                                  'weight': 'bold',
                                  'size': 7
                            })
        for text in leg.get_texts():
            plt.setp(text, color='#555555')
        return g

    def _pci_add_ci_level(self, g, df_ci, tau_list=None, alpha_lines=0.5,
                          y_axis='t_stat', emphasis='profile-likelihood'):
        '''
        Overlays confidence levels over the t-statistic (more intuitive)
        '''
        msg = ('Please set <y_axis> to either "t_stat", "f_stat", or "level" '
               'for plotting confidence interval curves.')
        assert y_axis in ['t_stat', 'f_stat', 'level'], msg
        if tau_list is None:
            if y_axis == 't_stat':
                tau_list = list(zip(df_ci[y_axis], df_ci['level']))
            elif y_axis == 'f_stat':
                tau_list = list(zip(df_ci[y_axis], df_ci['level']))
            else:
                tau_list = list(zip(df_ci[y_axis], df_ci['t_stat']))
        x_min, x_max = g.fig.axes[0].get_xlim()
        y_min, y_max = g.fig.axes[0].get_ylim()
        # to conform with _add_labels()
        x_max_new = ((x_max - x_min) * 1.1) + x_min  # adds 10% to right
        zorder = 0
        for idx, tau_pair in enumerate(tau_list):
            if emphasis == 'none':
                xmin_data = min(df_ci[df_ci[y_axis] ==
                                      tau_pair[0]]['wald_l'].item(),
                                df_ci[df_ci[y_axis] ==
                                      tau_pair[0]]['pl_l'].item(),
                                df_ci[df_ci[y_axis] ==
                                      tau_pair[0]]['boot_l'].item())
                x_val_l = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_l'].item()
                x_val_u = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_u'].item()
            elif emphasis == 'wald':
                xmin_data = df_ci[df_ci[y_axis] ==
                                  tau_pair[0]]['wald_l'].item()
                x_val_l = df_ci[df_ci[y_axis] == tau_pair[0]]['wald_l'].item()
                x_val_u = df_ci[df_ci[y_axis] == tau_pair[0]]['wald_u'].item()
            elif emphasis == 'profile-likelihood':
                xmin_data = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_l'].item()
                x_val_l = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_l'].item()
                x_val_u = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_u'].item()
            elif emphasis == 'bootstrap':
                xmin_data = df_ci[df_ci[y_axis] ==
                                  tau_pair[0]]['boot_l'].item()
                x_val_l = df_ci[df_ci[y_axis] == tau_pair[0]]['boot_l'].item()
                x_val_u = df_ci[df_ci[y_axis] == tau_pair[0]]['boot_u'].item()
            xmin = (xmin_data - x_min) / (x_max_new - x_min)

            if idx == 0:
                xmax = (x_max - x_min) / ((x_max_new*1.05) - x_min)
            else:
                xmax = (x_max - x_min) / (x_max_new - x_min)

            if y_axis == 'level':
                y_val = tau_pair[0] * 100
                ymax = (tau_pair[0]*100 - y_min) / (y_max - y_min)
            else:
                y_val = tau_pair[0]
                ymax = (tau_pair[0] - y_min) / (y_max - y_min)

            if len(tau_list) - idx <= 5:
                g.ax.axhline(y_val, xmin=xmin, xmax=xmax, linestyle='--',
                             linewidth=0.5, color='#7b7b7b',
                             label='{0:.2f}'.format(tau_pair[1]),
                             alpha=alpha_lines, zorder=zorder)
                zorder += 1
                g.ax.axvline(x_val_l, ymax=ymax, linestyle='--', linewidth=0.5,
                             color='#7b7b7b', alpha=alpha_lines, zorder=zorder)
                zorder += 1
                g.ax.axvline(x_val_u, ymax=ymax, linestyle='--', linewidth=0.5,
                             color='#7b7b7b', alpha=alpha_lines, zorder=zorder)
                zorder += 1
        return g

    def _pci_plot_emphasis(self, g, emphasis, df_ci, y_ci, lw_thick=1.5,
                           lw_thin=0.8):
        '''
        Draws confidence interval curves considering emphasis
        <emphasis> --> what curve to draw with empahsis
        <df_ci> --> the dataframe containing confidence interval data
        <y_ci> --> the y data to plot
        '''
        if emphasis.lower() == 'wald':
            g.ax.plot(df_ci['wald_l'], y_ci,
                      color=self.palette[3], linestyle='-', linewidth=lw_thick,
                      label='Wald', zorder=1)
            g.ax.plot(df_ci['wald_u'], y_ci,
                      color=self.palette[3], linestyle='-', linewidth=lw_thick,
                      label='wald_u', zorder=1)
        else:
            g.ax.plot(df_ci['wald_l'], y_ci,
                      color=self.palette[3], linestyle='-', linewidth=lw_thin,
                      label='Wald', zorder=1)
            g.ax.plot(df_ci['wald_u'], y_ci,
                      color=self.palette[3], linestyle='-', linewidth=lw_thin,
                      label='wald_u', zorder=1)
        if emphasis.lower() == 'profile-likelihood':
            g.ax.plot(df_ci['pl_l'], y_ci,
                      color=self.palette[2], linestyle='--',
                      linewidth=lw_thick,
                      label='Profile Likelihood', zorder=2)
            g.ax.plot(df_ci['pl_u'], y_ci,
                      color=self.palette[2], linestyle='--',
                      linewidth=lw_thick,
                      label='profile-likelihood_u', zorder=2)
        else:
            g.ax.plot(df_ci['pl_l'], y_ci,
                      color=self.palette[2], linestyle='--', linewidth=lw_thin,
                      label='Profile Likelihood', zorder=2)
            g.ax.plot(df_ci['pl_u'], y_ci,
                      color=self.palette[2], linestyle='--', linewidth=lw_thin,
                      label='profile-likelihood_u', zorder=2)
        if emphasis.lower() == 'bootstrap':
            g.ax.plot(df_ci['boot_l'], y_ci,
                      color=self.palette[0], linestyle='-.',
                      linewidth=lw_thick,
                      label='Bootstrap', zorder=3)
            g.ax.plot(df_ci['boot_u'], y_ci,
                      color=self.palette[0], linestyle='-.',
                      linewidth=lw_thick,
                      label='bootstrap_u', zorder=3)
        else:
            g.ax.plot(df_ci['boot_l'], y_ci,
                      color=self.palette[0], linestyle='-.',
                      linewidth=lw_thin,
                      label='Bootstrap', zorder=3)
            g.ax.plot(df_ci['boot_u'], y_ci,
                      color=self.palette[0], linestyle='-.', linewidth=lw_thin,
                      label='bootstrap_u', zorder=3)
        return g

    def _pci_add_labels(self, g, df_ci, tau_list=None, y_axis='t_stat'):
        '''
        Adds confidence interval or test statistic labels to the plot
        '''
        msg = ('Please set <y_axis> to either "t_stat", "f_stat", or "level" '
               'for plotting confidence interval curves.')
        assert y_axis in ['t_stat', 'f_stat', 'level'], msg
        if tau_list is None:
            if y_axis == 't_stat':
                tau_list = list(zip(df_ci[y_axis], df_ci['level']))
            elif y_axis == 'f_stat':
                tau_list = list(zip(df_ci[y_axis], df_ci['level']))
            elif y_axis == 'level':
                tau_list = list(zip(df_ci[y_axis]*100, df_ci['t_stat']))
        x_min, x_max = g.fig.axes[0].get_xlim()
        y_min, y_max = g.fig.axes[0].get_ylim()
    #    x_pos = self.eonr + (x_max - self.eonr)/2
#        x_max_new = ((x_max - x_min) * 1.1) - abs(x_min)  # adds 10% to right
        x_max_new = ((x_max - x_min) * 1.1) + x_min  # adds 10% to right
        x_pos = ((x_max - x_min) * 1.07) + x_min  # adds 10% to right
        g.fig.axes[0].set_xlim(right=x_max_new)
    #    boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
    #    el = mpatches.Ellipse((0, 0), 0.3, 0.3, angle=50, alpha=0.5)
    #    g.ax.add_artist(el)
        for idx, tau_pair in enumerate(tau_list):
            if y_axis == 'level':
                if idx == 0:
                    label = r'$|\tau|$ (T) = {0:.2f}'.format(tau_pair[1])
                    y_pos = tau_pair[0] + (y_max - y_min)*0.02
                else:
                    label = '{0:.2f}'.format(tau_pair[1])
            else:
                if idx == 0:
                    label = 'CI = {0:.0f}%'.format(tau_pair[1]*100)
                    y_pos = tau_pair[0] + (y_max - y_min)*0.02
                else:
                    label = '{0:.0f}%'.format(tau_pair[1]*100)
            if idx == 0:
                g.ax.annotate(label,
                              xy=(x_pos, y_pos),
                              xycoords='data',
                              fontsize=7, fontweight='light',
                              horizontalalignment='right',
                              verticalalignment='center',
                              color='#7b7b7b')
            elif len(tau_list) - idx <= 5:
                g.ax.annotate(label,
                              xy=(x_pos, tau_pair[0]),
                              xycoords='data',
                              fontsize=7, fontweight='light',
                              horizontalalignment='right',
                              verticalalignment='center',
                              color='#7b7b7b')
        return g

    def _set_title_size(self, g, title_text, size_font, coef=1.3):
        '''
        Sets the font size of a text object so it fits in the figure
        1.3 works pretty well for tau plots

        '''
        text_obj_dummy = plt.text(0, 0, title_text, size=size_font)
        x_min, x_max = g.fig.axes[0].get_xlim()
        dpi = g.fig.get_dpi()
        width_in, _ = g.fig.get_size_inches()
        pix_width = dpi * width_in
        r = g.fig.canvas.get_renderer()
        pix_width = r.width
        text_size = text_obj_dummy.get_window_extent(renderer=r)
        while (text_size.width * coef) > pix_width:
            size_font -= 1
            text_obj_dummy.set_size(size_font)
            text_size = text_obj_dummy.get_window_extent(renderer=r)
        text_obj_dummy.remove()
        return size_font

    def _pci_add_title(self, g, size_font=10):
        '''
        Adds the title to profile-liikelihood plot
        '''
        title_text = ('{0} {1} - {2} N Fertilizer Timing'
                      ''.format(self.year, self.location, self.time_n))
        divider = make_axes_locatable(g.ax)
        cax = divider.append_axes("top", size="15%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.set_facecolor('#7b7b7b')
        text_obj = AnchoredText(title_text, loc=10, frameon=False,
                                prop=dict(backgroundcolor='#7b7b7b',
                                          size=size_font, color='white',
                                          fontweight='bold'))
        size_font = self._set_title_size(g, title_text, size_font, coef=1.3)
        text_obj.prop.set_size(size_font)
        cax.add_artist(text_obj)

    def set_ratio(self, cost_n_fert=None, cost_n_social=None,
                  price_grain=None):
        '''
        Sets/resets the fertilizer cost, social cost of N, and price of grain

        Recomputes the price ratio based on this information, then adjusts the
        lowest level folder of base_dir to the ratio (e.g., \trad_0010\)
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

    def calculate_eonr(self, df, bootstrap_ci=True):
        '''
        Calculates EONR for <df> and saves results to self
        <n_steps> is the number of values across the range of N rates for which
        the gross return and cost curves will be calculated. As <n_steps>
        increases, the EONR calculation becomes more precise (recommended >
        300).
        '''
        self.bootstrap_ci = bootstrap_ci
        self._reset_temp()
        self._set_df(df)
        self._replace_missing_vals(missing_val='.')
        self._calc_grtn()
        if self.cost_n_social > 0:
            self._calc_social_cost(col_x=self.col_nup_soil_fert,
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
        results = [[self.price_grain, self.cost_n_fert, self.cost_n_social,
                    self.price_ratio, self.location, self.year, self.time_n,
                    self.coefs_grtn_primary['b0'].n, self.eonr,
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

    def print_results(self):
        '''
        Prints the coefficients comprising the Economic Optimum Nitrogen Rate:
            <mrtn>
            <eonr>
        '''
        self._print_results()

    def plot_eonr(self, x_min=None, x_max=None, y_min=None, y_max=None,
                  style='ggplot', ci_type='profile-likelihood', ci_level=None,
                  idx=None):
        '''
        Plot EONR, MRTN, GRTN, and N cost
        <style> can be any of the options supported by matplotlib:
        https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
        <ci_type> can be 'wald', 'profile-likelihood', or 'bootstrap':
            'wald' uses uniform conf. ints (not recommended for EONR data)
            'profile-likelihood' is based on work from Cook and Weisberg, 1990
                and is demonstrated for EONR by Hernandez and Mulla, 2008
            'bootstrap' uses a bootstraping method to explore the sampling
                distribution of the EONR to estimate the conf. ints
                (demonstrated by Hernandez and Mulla, 2008)
        <ci_level> can be set to manually adjust the confidence level to
            display on the plot (defaults to eonr.ci_level)
        '''
        self._set_style(style)
##        palette_blue = self._color_to_palette(color='b')
#        self.palette = self._seaborn_palette(color='muted', cmap_len=10)

        g = sns.FacetGrid(self.df_data)
        self._plot_points(self.col_n_app, 'grtn', self.df_data,
                          [self.palette[0]], ax=g.ax)
        if self.cost_n_social > 0:
            try:
                self._plot_points(self.col_nup_soil_fert, 'social_cost_n',
                                  self.df_data, [self.palette[8]], ax=g.ax)
            except KeyError as err:
                print('{0}\nFixed social cost of N was used, so points will '
                      'not be plotted..'.format(err))
        self._modify_axes_labels(fontsize=14)
        g = self._add_ci(g, ci_type=ci_type, ci_level=ci_level)
        g = self._draw_lines(g, self.palette)
        g = self._modify_minmax_axes(g, x_min=x_min, x_max=x_max,
                                     y_min=y_min, y_max=y_max)
        g = self._modify_axes_pos(g)
        g = self._place_legend(g, loc='upper left')
        g = self._modify_plot_params(g, plotsize_x=7, plotsize_y=4,
                                     labelsize=11)
        g = self._add_labels(g, x_max)
        g = self._add_title(g)
        plt.tight_layout()

        self.fig_eonr = g

    def plot_tau(self, y_axis='t_stat', emphasis='profile-likelihood',
                 run_n=None):
        '''
        Plots the t-statistic as a function of optimum nitrogen rate
        <y_axis> --> value to plot on the y-axis; change to 'level' to plot the
            confidence level as a function of EOR instead of tau
        '''
        if run_n is None:
            df_ci = self.df_ci[self.df_ci['run_n'] ==
                               self.df_ci['run_n'].max()]
        else:
            df_ci = self.df_ci[self.df_ci['run_n'] == run_n]
        y_axis = str(y_axis).lower()
        emphasis = str(emphasis).lower()
        msg1 = ('Please set <y_axis> to either "t_stat", "f_stat", or "level" '
                'for plotting confidence interval curves.')
        assert y_axis in ['t_stat', 'f_stat', 'level'], msg1
        msg2 = ('Please set <emphasis> to either "wald", '
                '"profile-likelihood", "bootstrap", or "None" to indicate '
                'which curve (if any) should be emphasized.')
        assert emphasis in ['wald', 'profile-likelihood', 'bootstrap', 'None',
                            None], msg2
        g = sns.FacetGrid(df_ci)
        if y_axis == 'level':
            y_ci = df_ci[y_axis] * 100
        else:
            y_ci = df_ci[y_axis]
        g = self._pci_plot_emphasis(g, emphasis, df_ci, y_ci)

        if emphasis == 'wald':
            self._plot_points(col_x='wald_l', col_y=y_ci, data=df_ci,
                              palette=self.palette[3], ax=g.ax, s=20,
                              zorder=4)
            self._plot_points(col_x='wald_u', col_y=y_ci, data=df_ci,
                              palette=self.palette[3], ax=g.ax, s=20,
                              zorder=4)
        elif emphasis == 'profile-likelihood':
            self._plot_points(col_x='pl_l', col_y=y_ci, data=df_ci,
                              palette=self.palette[2], ax=g.ax, s=20,
                              zorder=4)
            self._plot_points(col_x='pl_u', col_y=y_ci, data=df_ci,
                              palette=self.palette[2], ax=g.ax, s=20,
                              zorder=4)
        elif emphasis == 'bootstrap':
            self._plot_points(col_x='boot_l', col_y=y_ci, data=df_ci,
                              palette=self.palette[0], ax=g.ax, s=20,
                              zorder=4)
            self._plot_points(col_x='boot_u', col_y=y_ci, data=df_ci,
                              palette=self.palette[0], ax=g.ax, s=20,
                              zorder=4)

        self._pci_modify_axes_labels(y_axis)
        g = self._pci_place_legend_tau(g)
        g = self._modify_plot_params(g, plotsize_x=4.5, plotsize_y=4,
                                     labelsize=11)
        g.ax.tick_params(axis='both', labelsize=9)
        g = self._pci_add_ci_level(g, df_ci, y_axis=y_axis, emphasis=emphasis)
        g = self._pci_add_labels(g, df_ci, y_axis=y_axis)
        self._pci_add_title(g)
        g.ax.grid(False, axis='x')
        y_grids = g.ax.get_ygridlines()
        for idx, grid in enumerate(y_grids):
            if idx == 1:
                grid.set_linewidth(1.0)
            else:
                grid.set_alpha(0.4)
        plt.tight_layout()
        self.fig_tau = g

    def save_plot(self, base_dir=None, fname='eonr_sns17w_pre.png', fig=None,
                  dpi=300):
        if fig is None:
            fig = self.fig_eonr
        msg = ('A figure must be generated first. Please execute '
               'EONR.plot_eonr() first..\n')
        assert fig is not None, msg
        if base_dir is None:
            base_dir = self.base_dir
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
        fname = os.path.join(base_dir, fname)
        fig.savefig(fname, dpi=dpi)

    def eonr_delta(self, df_results=None):
        '''
        Inserts a new column "eonr_delta" into df_results. All data is filtered
        by location, year, and N timing, then "eonr_delta" is calculated as the
        difference from the economic scenario resulting in the highest EONR
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

    def plot_profile_likelihood(self, df_ci=None, n_rows=3):
        '''
        Plots the profile likelihood confidence curve for a range of alpha
        values (see Cook and Weisberg, 1990)
        '''
        if df_ci is None:
            df_ci = self.df_ci
#        def convert_axis(df_ci, ax, ax2):
#            ax2.set_ylim(df_ci['level'].iloc[0], df_ci['level'].iloc[-1])
#            ax2.figure.canvas.draw()
#            ax2.grid(False)
        n_datasets = len(df_ci.drop_duplicates(subset=['year', 'location',
                                                       'time_n']))
        n_ratios = len(df_ci['price_ratio'].unique())

        years = df_ci['year'].unique()
        years.sort()
        for year in years:
            df_year = df_ci[df_ci['year'] == year]
            locs = df_year['location'].unique()
            locs.sort()
            for loc in locs:
                df_loc = df_year[df_year['location'] == loc]
                times = df_loc['time_n'].unique()
                for time in times:
                    df_yloct = df_loc[df_loc['time_n'] == time]
                    ratios = df_yloct['price_ratio'].unique()
                    count = 0
                    fig, ax = plt.subplots(1, len(ratios), sharey=True, figsize=(20,3))
                    for ratio in ratios:
                        df_filt = df_yloct[df_yloct['price_ratio'] == ratio]
                #        ax2 = plt.twinx()
                #        ax.callbacks.connect("ylim_changed", convert_axis(df_ci, ax, ax2))
                        sns.lineplot(df_filt['wald_l'], df_filt['t_stat'], ax=ax[count], color='#7b7b7b')
                        sns.lineplot(df_filt['wald_u'], df_filt['t_stat'], ax=ax[count], color='#7b7b7b')
                        sns.scatterplot(df_filt['pl_l'], df_filt['t_stat'], ax=ax[count], color='red')
                        sns.scatterplot(df_filt['pl_u'], df_filt['t_stat'], ax=ax[count], color='red')
                        sns.scatterplot(df_filt['boot_l'], df_filt['t_stat'], ax=ax[count], color='blue')
                        sns.scatterplot(df_filt['boot_u'], df_filt['t_stat'], ax=ax[count], color='blue')
                        ax[count].set_xlabel('EONR')
#                        ax[count].set_ylabel('T-statistic')
                        ax[count].set_title('{0} {1} {2} - {3}'.format(year, loc, time, ratio), fontsize=8)
                        count += 1
#                        col += 1
#                        print((((row*n_cols)+n_cols) % count + 1))
#                        if (((row*n_cols)+n_cols) % count + 1) == 0:
#                            row += 1
#                            col = 0
#                        break
#                    input("Press Enter to continue...")
                    ax[0].set_ylabel('T-statistic')
                    plt.tight_layout()
                    break
                break
            break

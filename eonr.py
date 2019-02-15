# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 20:15:17 2019

@author: nigo0024

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
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
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

        self.figure = None
        self.palette = None
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
                                                'eonr', 'ci_level',
                                                'ci_wald_l', 'ci_wald_u',
                                                'ci_pl_l', 'ci_pl_u',
                                                'ci_boot_l', 'ci_boot_u',
                                                'mrtn', 'grtn_r2_adj',
                                                'grtn_max_y', 'grtn_crit_x',
                                                'grtn_y_int', 'scn_lin_r2',
                                                'scn_lin_rmse', 'scn_exp_r2',
                                                'scn_exp_rmse'])
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
        print('\nComputing EONR for {0} {1} {2}\nCost of N fertilizer: '
              '{3}{4:.2f} per {5}\nPrice grain: {6}{7:.2f} per {8}'
              ''.format(self.location, self.year, self.time_n,
                        self.unit_currency, self.cost_n_fert, self.unit_fert,
                        self.unit_currency, self.price_grain, self.unit_grain))
        if self.cost_n_social > 0:
            print('Social cost of N: {0}{1:.2f} \n'
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

    def _f_quad_plateau(self, x, a, b, c):
        '''
        Quadratic plateau function

        Y =
        beta0 + beta1*x +beta2*(x^2) + error
            --> if x < <crit_x>
        beta0 - (beta1^2)/(4*beta2) + error
            --> if x >= <crit_x>
        where crit_x = -beta1/(2*beta2)
        '''
        crit_x = -b/(2*c)
        if isinstance(x, int) or isinstance(x, float):
            y = 0
            y += (a + b*x + c*(x**2)) * (x < crit_x)
            y += (a - (b**2) / (4*c)) * (x >= crit_x)
            return y
        else:
            array_temp = np.zeros(len(x))
            array_temp += (a + b*x + c*(x**2)) * (x < crit_x)
            array_temp += (a - (b**2) / (4*c)) * (x >= crit_x)
            return array_temp

    def _f_qp_theta2(self, x, theta11, theta2, theta12):
        '''
        Quadratic plateau function using theta2 (EOR) as one of the parameters

        theta2 = (R-beta1)/(2*theta12) = EOR
            where R is the price ratio and theta12 = beta2
        rearrange to get:

        beta1_hat = -(2*theta2*theta12-R)

        beta1_hat can replace beta1 in the original quadratic plateau model to
        incorporate theta2 (EOR) as one of the parameters. This is desireable
        because we can use scipy.optimize.curve_fit() to compute the covariance
        matrix and estimate confidence intervals for theta2 directly.
        '''
        R = self.R
        beta1 = -(2*theta2*theta12 - R)

        if isinstance(x, int) or isinstance(x, float):
            y = 0
            y += (theta11 + beta1*x + theta12*(x**2)) * (x < theta2)
            y += (theta11 - (beta1**2) / (4*theta12)) * (x >= theta2)
            return y
        else:
            array_temp = np.zeros(len(x))
            array_temp += (theta11 + beta1*x + theta12*(x**2)) * (x < theta2)
            array_temp += (theta11 - (beta1**2) / (4*theta12)) * (x >= theta2)
            return array_temp

    def _f_qp_theta2_social(self, x, theta11, theta2, theta12):
        '''
        Same as _f_qp_theta2(), except sets R because best fit was
        already computed assuming R to be equal to self.R
        '''
        R = self.R
        crit_x = self.coefs_grtn['crit_x']
        beta1 = -(2*theta2*theta12 - R)

        if isinstance(x, int) or isinstance(x, float):
            y = 0
            y += (theta11 + beta1*x + theta12*(x**2)) * (x < crit_x)
            y += (theta11 - (beta1**2) / (4*theta12)) * (x >= crit_x)
            return y
        else:
            array_temp = np.zeros(len(x))
            array_temp += (theta11 + beta1*x + theta12*(x**2)) * (x < crit_x)
            array_temp += (theta11 - (beta1**2) / (4*theta12)) * (x >= crit_x)
            return array_temp

    def _f_combine_exp(self, x, theta11, theta12):
        # Additions
        theta2 = self.ci_theta2

        exp_a = self.coefs_social['exp_a']
        exp_b = self.coefs_social['exp_b']
        exp_c = self.coefs_social['exp_c']
        social_cost = self._f_exp(x, a=exp_a, b=exp_b, c=exp_c)
        beta0 = theta11 - social_cost

        price_ratio = self.price_ratio - (self.cost_n_social /
                                          self.price_grain)
        R = price_ratio * self.price_grain

        beta1 = -(2*theta2*theta12 - R)

        if isinstance(x, int) or isinstance(x, float):
            y = 0
            y += (beta0 + beta1*x + theta12*(x**2)) * (x < theta2)
            y += (beta0 - (beta1**2) / (4*theta12)) * (x >= theta2)
            return y
        else:
            array_temp = np.zeros(len(x))
            array_temp += (beta0 + beta1*x + theta12*(x**2)) * (x < theta2)
            array_temp += (beta0 - (beta1**2) / (4*theta12)) * (x >= theta2)
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
        qp_a = self.coefs_grtn['coef_a'].n
        qp_b = self.coefs_grtn['coef_b'].n
        qp_c = self.coefs_grtn['coef_c'].n
        gross_rtn = self._f_quad_plateau(x, a=qp_a, b=qp_b, c=qp_c)
        # Subtractions
        fert_cost = x * self.cost_n_fert

        exp_a = self.coefs_social['exp_a']
        exp_b = self.coefs_social['exp_b']
        exp_c = self.coefs_social['exp_c']
        social_cost = self._f_exp(x, a=exp_a, b=exp_b, c=exp_c)

        result = gross_rtn - fert_cost - social_cost
        return -result

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
            exp_r2, _, _, _, exp_rmse = self._get_rsq(self._f_exp, x, y, popt)
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
            except OptimizeWarning as err:
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
            return popt, pcov
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    popt, pcov = curve_fit(f, xdata, ydata, p0=p0,
                                           maxfev=maxfev)
                    return popt, pcov
                except OptimizeWarning as err:
                    if self.print_out is True:
                        print('Information for which the OptimizeWarning was '
                              'thrown:\n{0}'.format(info))
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
            theta12 = self.coefs_grtn['coef_c'].n
            b = self.coefs_grtn['coef_b'].n
            self.R = b + (2 * theta12 * self.eonr)  # as a starting point

            guess = (self.coefs_grtn['coef_a'].n,
                     self.coefs_grtn['crit_x'],
                     self.coefs_grtn['coef_c'].n)

            popt, pcov = self._curve_fit_runtime(self._f_qp_theta2_social, x,
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
                popt, pcov = self._curve_fit_runtime(self._f_qp_theta2_social,
                                                     x, y, guess, maxfev=800)
                dif = abs(popt[1] - self.eonr)
            res = y - self._f_qp_theta2(x, *popt)
            ss_res = np.sum(res**2)
            if (popt is None or np.any(popt == np.inf) or
                    np.any(pcov == np.inf)):
                theta11 = unc.ufloat(popt[0], 0)
                theta2 = unc.ufloat(popt[1], 0)
                theta12 = unc.ufloat(popt[2], 0)
            else:
                theta11, theta2, theta12 = unc.correlated_values(popt, pcov)
            info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                    ''.format('_compute_R() -> _f_quad_plateau', col_x,
                              col_y + ' (residuals)'))
            popt, pcov = self._curve_fit_opt(lambda x, b: self._f_quad_plateau(
                    x, theta11.n, b, theta12.n), x, y, info=info)
            popt = np.insert(popt, 0, theta12.n)
            popt = np.insert(popt, 2, theta11.n)
            f = np.poly1d(popt)
            result = minimize_scalar(-f)

            self.coefs_nrtn['theta11'] = theta11
            self.coefs_nrtn['theta2'] = theta2
            self.coefs_nrtn['theta12'] = theta12
            self.coefs_nrtn['theta2_social'] = result['x']
            self.coefs_nrtn['popt_social'] = [popt[2],
                                              minimize_scalar(-f)['x'],
                                              popt[0]]
            self.coefs_nrtn['ss_res_social'] = ss_res

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
        ss_res = np.sum(res**2)
        rmse = (ss_res)**0.5
        y_mean = np.mean(y)
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
        y_max = (self.coefs_grtn['coef_a'].n +
                 (self.coefs_grtn['crit_x'] *
                  self.coefs_grtn['coef_b'].n) +
                 (self.coefs_grtn['crit_x'] *
                  self.coefs_grtn['crit_x'] *
                  self.coefs_grtn['coef_c'].n))
        # Find index where all x = max
        y_temp = (self.coefs_grtn['coef_a'].n +
                  (x1*self.coefs_grtn['coef_b'].n) +
                  (x1*x1*self.coefs_grtn['coef_c'].n))
        y_max_idx = np.argmax(y_temp)
        y2a = (self.coefs_grtn['coef_a'].n +
               (x1a[:y_max_idx]*self.coefs_grtn['coef_b'].n) +
               (x1a[:y_max_idx]*x1a[:y_max_idx]*self.coefs_grtn['coef_c'].n))
        y2b = np.linspace(y_max, y_max, num=n_steps-y_max_idx)
        y_grtn = np.concatenate((y2a, y2b))

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
        self.results_temp['grtn_y_int'] = self.coefs_grtn['coef_a'].n

        if self.base_zero is True:
            self.df_data['grtn'] = (self.df_data['grtn'] -
                                    self.coefs_grtn['coef_a'].n)
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
        by the price of grain to get into units of lbs N, which is a unit
        that can be used with the price ratio R. When dividing by grain price,
        this value can be thought of as the N rate required to increase yield
        by 1 bushel (or the other way around??). Let's call it sn_ratio.

        Should we then multiply the sn_ratio by the price ratio before shifting
        theta2? Try both..
        '''
        df_data = self.df_data.copy()
        x = df_data[col_x].values
        y = df_data[col_y].values

        guess = (self.coefs_grtn['coef_a'].n,
                 self.coefs_grtn['crit_x'],
                 self.coefs_grtn['coef_c'].n)
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_calc_nrtn() -> _f_qp_theta2',
                          col_x, col_y))
        popt, pcov = self._curve_fit_opt(self._f_qp_theta2, x, y,
                                         p0=guess, maxfev=1000, info=info)
        res = y - self._f_qp_theta2(x, *popt)
        # if cost_n_social > 0, this will be dif than coefs_grtn['ss_res']
        ss_res = np.sum(res**2)

        if popt is None or np.any(popt == np.inf) or np.any(pcov == np.inf):
            theta11 = unc.ufloat(popt[0], 0)
            theta2 = unc.ufloat(popt[1], 0)
            theta12 = unc.ufloat(popt[2], 0)
        else:
            theta11, theta2, theta12 = unc.correlated_values(popt, pcov)

        self.coefs_nrtn = {
                'theta11': theta11,
                'theta2': theta2,
                'theta12': theta12,
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
                self.coefs_grtn['coef_a'].n,
                self.coefs_grtn['coef_b'].n,
                self.coefs_grtn['coef_c'].n))
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
            boot_l = self.df_ci_temp['boot_l'].item()
            boot_u = self.df_ci_temp['boot_u'].item()
        except TypeError as err:
            print(err)

        print('\nEconomic optimum N rate (EONR): {0:.1f} {1} [{2:.1f}, '
              '{3:.1f}] ({4:.1f}% confidence)'
              ''.format(self.eonr, self.unit_nrate, pl_l, pl_u,
                        (self.ci_level*100)))
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
            beta0 = unc.ufloat(popt[0], 0)
            beta1 = unc.ufloat(popt[1], 0)
            beta2 = unc.ufloat(popt[2], 0)
        else:
            beta0, beta1, beta2 = unc.correlated_values(popt, pcov)

        crit_x = -beta1.n/(2*beta2.n)
        max_y = self._f_quad_plateau(crit_x, beta0.n, beta1.n, beta2.n)
        r2, r2_adj, ss_res, ss_tot, rmse = self._get_rsq(self._f_quad_plateau,
                                                         x, y, popt)
        aic = self._calc_aic(x, y, dist='gamma')

        if rerun is False:
            self.coefs_grtn = {
                    'coef_a': beta0,
                    'coef_b': beta1,
                    'coef_c': beta2,
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
                    'theta11': beta0,
                    'theta2': None,
                    'theta12': beta2,
                    'popt': None,
                    'pcov': None,
                    'ss_res': None,
                    'theta2_social': None,
                    'popt_social': None,
                    'ss_res_social': None
                    }
        else:
            self.coefs_grtn_primary = self.coefs_grtn.copy()
            self.coefs_grtn = {}
            self.coefs_grtn = {
                    'coef_a': beta0,
                    'coef_b': beta1,
                    'coef_c': beta2,
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
        y_max = (self.coefs_grtn['coef_a'].n +
                 (self.coefs_grtn['crit_x'] * self.coefs_grtn['coef_b'].n) +
                 (self.coefs_grtn['crit_x'] * self.coefs_grtn['crit_x'] *
                  self.coefs_grtn['coef_c'].n))
        # Find index where all x = max
        y_temp = (self.coefs_grtn['coef_a'].n +
                  (x1*self.coefs_grtn['coef_b'].n) +
                  (x1*x1*self.coefs_grtn['coef_c'].n))
        y_max_idx = np.argmax(y_temp)
        y2a = (self.coefs_grtn['coef_a'].n +
               (x1a[:y_max_idx]*self.coefs_grtn['coef_b'].n) +
               (x1a[:y_max_idx]*x1a[:y_max_idx]*self.coefs_grtn['coef_c'].n))
        y2b = np.linspace(y_max, y_max, num=n_steps-y_max_idx)
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
        f_eonr1 = np.poly1d([self.coefs_nrtn['theta12'].n,
                            self.coefs_nrtn['theta2'].n,
                            self.coefs_nrtn['theta11'].n])
        f_eonr2 = np.poly1d([self.coefs_grtn['coef_c'].n,
                            self.coefs_grtn['coef_b'].n,
                            self.coefs_grtn['coef_a'].n])
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
                result = minimize_scalar(self._f_combine_rtn_cost,
                                         bounds=[-100, 365], method='bounded')
        else:
            first_order = self.coefs_grtn['coef_b'].n - self.cost_n_fert
            f_eonr2 = self._modify_poly1d(f_eonr2, 1, first_order)
            result = minimize_scalar(-f_eonr2)
        # theta2 is EOR (minimum) only if total cost of N increases linearly
        # at a first order with an intercept of zero..
        self.eonr = result['x']
        self.mrtn = -result['fun']

    #  Following are functions used in calculating confidence intervals
    def _bs_statfunction(self, x, y):
        '''
        '''
        maxfev = 1000
        a = self.coefs_grtn['coef_a'].n
        c = self.coefs_grtn['coef_c'].n
        guess = (a, self.eonr, c)
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
                                    self.cost_n_fert,
                                    self.cost_n_social,
                                    0, 0, 0, self.eonr,
                                    self.eonr, self.eonr, self.eonr]],
                             columns=['location', 'year', 'time_n',
                                      'cost_n_fert', 'cost_n_social',
                                      'f_stat', 't_stat', 'level',
                                      'wald_l', 'wald_u',
                                      'pl_l', 'pl_u'])
        return df_ci

    def _calc_sse_full(self, x, y):
        '''
        Calculates the sum of squares across the full set of parameters,
        solving for theta2
        '''
        guess = (self.coefs_grtn['coef_a'].n,
                 self.coefs_grtn['crit_x'],
                 self.coefs_grtn['coef_c'].n)
        col_x = None  # Perhaps we should keep in col_x/ytil curve_fit runs..?
        col_y = None
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_calc_sse_full() -> _f_qp_theta2_social',
                          col_x, col_y))
        if self.cost_n_social > 0:
            popt, pcov = self._curve_fit_opt(self._f_qp_theta2_social, x, y,
                                             p0=guess, info=info)
            res = y - self._f_qp_theta2_social(x, *popt)
            sse_full_theta2 = np.sum(res**2)
            self.coefs_nrtn['ss_res_social'] = sse_full_theta2
        else:
            popt, pcov = self._curve_fit_opt(self._f_qp_theta2, x, y,
                                             p0=guess, info=info)
            res = y - self._f_qp_theta2(x, *popt)
            sse_full_theta2 = np.sum(res**2)
        return sse_full_theta2

    def _check_progress_pl(self, step_size, tau_temp, tau, f_stat,
                           step_size_start, tau_delta_f, count):
        '''
        Checks progress of profile likelihood function and adjusts step
        size accordingly. Without this function, there is a chance some
        datasets will take millions of tries if the ci is wide
        '''
        tau_delta = tau_temp - tau
        progress = tau_temp / f_stat
        if progress < 0.9 and tau_delta < 1e-3:
            step_size = step_size_start
            step_size *= 1000
            tau_delta_f = True
        elif progress < 0.9 and tau_delta < 1e-2:
            step_size = step_size_start
            step_size *= 100
            tau_delta_f = True
        elif progress < 0.95 and count > 100:
            step_size = step_size_start
            step_size *= 1000
            tau_delta_f = True
        elif progress < 0.9:
            step_size = step_size_start
            step_size *= 10
            tau_delta_f = True
        else:
            if tau_delta_f is True and count < 100:
                step_size = step_size_start
            # else keep step size the same
        return step_size, tau_delta_f

    def _compute_bootstrap(self, alpha=0.1, n_samples=9999):
        '''
        Uses bootstrapping to estimate EONR based on the sampling distribution
        0) Initialize vector x with residuals - mean
        1) Sample with replacement from vector x (n = number of initial
             samples..)
        2) Calculate the percentile and "bias corrected and accelerated"
             bootstrapped CIs
        '''
        x = self.df_data[self.col_n_app].values
#        res = self.df_data['grtn_res'].values
#        boot_ci = bootstrap.ci((x, res), statfunction=self._bs_statfunction,
#                               alpha=alpha, n_samples=n_samples, method='bca')
        y = self.df_data['grtn'].values
        a = self.coefs_grtn['coef_a'].n
        c = self.coefs_grtn['coef_c'].n
        guess = (a, self.eonr, c)
        col_x = self.col_n_app
        col_y = 'grtn'
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format('_compute_bootstrap() -> _f_qp_theta2',
                          col_x, col_y))
        popt, _ = self._curve_fit_opt(self._f_qp_theta2, x, y,
                                      p0=guess, maxfev=1000, info=info)
        boot_ci = bootstrap.ci((x, y), statfunction=self._bs_statfunction,
                               alpha=alpha, n_samples=9999, method='bca')
        return boot_ci

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

    def _get_likelihood(self, theta2_start, alpha, x, y, side, tau_start=0,
                        step_size=1, epsilon=1e-9, df=None, sse_full=None):
        '''
        Computes the profile liklihood confidence values using the sum of
        squares (see Gallant (1987), p. 107)
        <theta2_start>: starting point of theta2 (should be set to maximum
            likelihood value/self.eonr)
        <alpha>: the significance level to compute the likelihood for
        <x> and <y>: the x and y data that likelihood should computed for
        <side>: The side of the confidence interval to compute the likelihood
            for - should be either "upper" or "lower" (dtype = str)
        '''
        msg = ('Please choose either "upper" or "lower" for <side> of '
               'confidence interval to compute.')
        assert side.lower() in ['upper', 'lower'], msg
        guess = (self.coefs_grtn['coef_a'].n,
                 self.coefs_grtn['coef_b'].n,
                 self.coefs_grtn['coef_c'].n)
        q = 1
        n = len(x)
        p = len(guess)
        f_stat = stats.f.ppf(1-alpha, dfn=q, dfd=n-p)
        if df is None:
            df = pd.DataFrame(columns=['theta', 'f_stat'])
        if sse_full is None:
            sse_full = self._calc_sse_full(x, y)
        s2 = sse_full / (n - p)
        theta2 = theta2_start
        tau = tau_start
        step_size_start = step_size
        tau_delta_f = False
        stop_flag = False
        count = 0
        col_x = self.col_n_app
        col_y = 'grtn'
        if self.cost_n_social > 0:
            str_func = '_get_likelihood() -> _f_qp_theta2_social'
        else:
            str_func = '_get_likelihood() -> _f_qp_theta2'
        info = ('func = {0}\ncol_x = {1}\ncol_y = {2}\n'
                ''.format(str_func, col_x, col_y))
        while tau < f_stat:
            if self.cost_n_social > 0:
                popt, pcov = self._curve_fit_runtime(
                        lambda x, theta11, theta12: self._f_qp_theta2_social(
                                x, theta11, theta2, theta12), x, y,
                        guess=(1, 1), maxfev=800, info=info)
            else:
                popt, pcov = self._curve_fit_runtime(
                        lambda x, theta11, theta12: self._f_qp_theta2(
                                x, theta11, theta2, theta12), x, y,
                        guess=(1, 1), maxfev=800, info=None)

            if popt is not None:
                popt = np.insert(popt, 1, theta2)
                if self.cost_n_social > 0:
                    res = y - self._f_qp_theta2_social(x, *popt)
                else:
                    res = y - self._f_qp_theta2(x, *popt)
                sse_res = np.sum(res**2)
                tau_temp = ((sse_res - sse_full) / q) / s2

            if count >= 1000:
                print('{0:.1f}% {1} profile likelihood failed to converge '
                      'after {2} iterations. Stopping calculation and using '
                      '{3:.1f} {4}'.format((1-alpha)*100, side.capitalize(),
                                           count, theta2, self.unit_nrate))
                tau = f_stat
                stop_flag = True
                theta_same = df['theta'].iloc[-1]
                df2 = pd.DataFrame([[theta_same, f_stat]],
                                   columns=['theta', 'f_stat'])
                df = df.append(df2, ignore_index=True)
                break  # break out of while loop
            elif popt is None:
                print('{0:.1f}% {1} profile likelihood failed to find optimal '
                      'parameters. Stopping calculation.'
                      ''.format((1-alpha)*100, side.capitalize()))
                tau = f_stat
                stop_flag = True
                if df['theta'].iloc[-1] is not None:
                    theta_rec = df['theta'].iloc[-1]
                    if theta_rec < 1.0:
                        theta_rec = 0
                else:
                    theta_rec = np.NaN
                df2 = pd.DataFrame([[theta_rec, f_stat]],
                                   columns=['theta', 'f_stat'])
                df = df.append(df2, ignore_index=True)
                break  # break out of while loop
            else:
                df2 = pd.DataFrame([[theta2, tau_temp]],
                                   columns=['theta', 'f_stat'])
                df = df.append(df2, ignore_index=True)

            count += 1
            step_size, tau_delta_f = self._check_progress_pl(
                    step_size, tau_temp, tau, f_stat, step_size_start,
                    tau_delta_f, count)
            if side.lower() == 'upper':
                theta2 += step_size
            elif side.lower() == 'lower':
                theta2 -= step_size
            tau = tau_temp

        if len(df) <= 1:
            # start over, reducing step size
            _, _, wald_l, wald_u, df = self._get_likelihood(
                    theta2_start, alpha, x, y, side, tau_start=tau_start,
                    step_size=(step_size/10), epsilon=epsilon, df=df,
                    sse_full=sse_full)
        elif stop_flag is True:  # If we had to stop calculation..
            wald_l, wald_u = self._compute_wald(n, p, alpha)
        elif abs(df['theta'].iloc[-1] - df['theta'].iloc[-2]) > epsilon:
            df = df[:-1]
            # At this point, we could get stuck in a loop if we never make any
            # headway on closing in on epsilon
            _, _, wald_l, wald_u, df = self._get_likelihood(
                    df['theta'].iloc[-1], alpha, x, y, side,
                    tau_start=df['f_stat'].iloc[-1],
                    step_size=(step_size/10),
                    epsilon=epsilon, df=df, sse_full=sse_full)
        else:
            wald_l, wald_u = self._compute_wald(n, p, alpha)
#            boot_l, boot_u = self._compute_bootstrap(alpha, n_samples=5000)
#        theta2_out = df['theta'].iloc[-2:].mean()
#        tau_out = df['f_stat'].iloc[-2:].mean()
        theta2_out = df['theta'].iloc[-1]
        tau_out = df['f_stat'].iloc[-1]
        return theta2_out, tau_out, wald_l, wald_u, df

    def _handle_no_ci(self):
        '''
        If critical x is greater than the max N rate, don't calculate
        confidence intervals, but fill out df_ci with Wald CIs only
        '''
        df_ci = self._build_df_ci()
        guess = (self.coefs_grtn['coef_a'].n,
                 self.coefs_grtn['coef_b'].n,
                 self.coefs_grtn['coef_c'].n)
        for alpha in self.alpha_list:
            level = 1 - alpha
            n = len(self.df_data[self.col_n_app])
            p = len(guess)
            wald_l, wald_u = self._compute_wald(n, p, alpha)
            t_stat = stats.t.ppf(level, len(self.df_data) - 3)
            df_row = pd.DataFrame([[self.df_data.iloc[0]['location'],
                                    self.df_data.iloc[0]['year'],
                                    self.df_data.iloc[0]['time_n'],
                                    self.cost_n_fert,
                                    self.cost_n_social,
                                    np.nan, t_stat, level,
                                    wald_l, wald_u, np.nan, np.nan]],
                                  columns=df_ci.columns)
            df_ci = df_ci.append(df_row, ignore_index=True)
        boot_ci = [None] * ((len(self.ci_list) * 2) + 2)
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
            boot_ci = [None] * ((len(self.ci_list) * 2) + 2)
        boot_ci = [self.eonr, self.eonr] + list(boot_ci)
        df_boot = self._parse_boot_ci(boot_ci)
        df_ci = pd.concat([df_ci, df_boot], axis=1)
        return df_ci

    #  Following are plotting functions
    def _add_labels(self, g):
        '''
        Adds EONR and economic labels to the plot
        '''
        boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
        el = mpatches.Ellipse((0, 0), 0.3, 0.3, angle=50, alpha=0.5)
        g.ax.add_artist(el)

        label_eonr = 'EONR: {0:.0f} {1}\nMRTN: ${2:.2f}'.format(
                self.eonr, self.unit_nrate, self.mrtn)
        g.ax.plot([self.eonr], [self.mrtn], marker='.', markersize=15,
                  color=self.palette[2], markeredgecolor='white',
                  label=label_eonr)
        if self.eonr > 0:
            g.ax.annotate(
                label_eonr,
                xy=(self.eonr, self.mrtn), xytext=(-80, -30),
                textcoords='offset points', ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5), alpha=0.9),
                arrowprops=dict(arrowstyle='-|>',
                                color="grey",
                                patchB=el,
                                shrinkB=10,
                                connectionstyle='arc3,rad=-0.3'))
        else:
            g.ax.annotate(
                label_eonr,
                xy=(0, 0), xycoords='axes fraction', xytext=(0.98, 0.05),
                ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5), alpha=0.9))

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
                    self.coefs_grtn_primary['coef_a'].n))
        g.ax.annotate(
            label_econ,
            xy=(0, 1), xycoords='axes fraction', xytext=(0.98, 0.95),
            horizontalalignment='right', verticalalignment='top',
            fontsize=7,
            bbox=dict(boxstyle=boxstyle_str,
                      fc=(1, 1, 1), ec=(0.5, 0.5, 0.5), alpha=0.75))
        return g

    def _add_title(self, g):
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
        at = AnchoredText(title_text, loc=10, frameon=False,
                          prop=dict(backgroundcolor='#7b7b7b',
                                    size=12, color='white',
                                    fontweight='bold'))
        cax.add_artist(at)
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
        g.ax.axvline(ci_l, linestyle='-', linewidth=0.5, color='#7b7b7b',
                     label=label_ci)
        g.ax.axvline(ci_u, linestyle='-', linewidth=0.5, color='#7b7b7b',
                     label=None)
        g.ax.axvline(self.eonr,
                     linestyle='--',
                     linewidth=1.5,
                     color='k',
                     label='EONR')
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

    def _draw_n_cost(self, g, palette):
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

    def _modify_minmax_axes(self, g, x_min, x_max, y_min, y_max):
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

    def _place_legend(self, g, loc='lower right', facecolor='white'):
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
        g.ax.legend(handles=handles,
                    labels=labels,
                    loc='upper left',
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
        return g

    def _plot_points(self, col_x, col_y, palette, ax=None, s=20):
        '''
        Plots points on figure
        '''
        if ax is None:
            g = sns.scatterplot(x=col_x,
                                y=col_y,
                                color=palette,
                                legend=False,
                                data=self.df_data,
                                s=s)
        else:
            g = sns.scatterplot(x=col_x,
                                y=col_y,
                                color=palette,
                                legend=False,
                                data=self.df_data,
                                ax=ax,
                                s=s)
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

    def calculate_eonr(self, df):
        '''
        Calculates EONR for <df> and saves results to self
        <n_steps> is the number of values across the range of N rates for which
        the gross return and cost curves will be calculated. As <n_steps>
        increases, the EONR calculation becomes more precise (recommended >
        300).
        '''
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
        if self.eonr > self.df_data[self.col_n_app].max():
            print('\nEONR is past the point of available data, so confidence '
                  'bounds are not being computed')
            self._handle_no_ci()
        else:
            self._compute_residuals()
            self._compute_cis(col_x=self.col_n_app, col_y='grtn')
        self._build_mrtn_lines()
        if self.print_out is True:
            self._print_grtn()
        self._print_results()
        results = [[self.price_grain, self.cost_n_fert, self.cost_n_social,
                    self.price_ratio, self.location, self.year, self.time_n,
                    self.eonr, self.ci_level,
                    self.df_ci_temp['wald_l'].item(),
                    self.df_ci_temp['wald_u'].item(),
                    self.df_ci_temp['pl_l'].item(),
                    self.df_ci_temp['pl_u'].item(),
                    self.df_ci_temp['boot_l'].item(),
                    self.df_ci_temp['boot_u'].item(),
                    self.mrtn, self.coefs_grtn['r2_adj'],
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
                  style='ggplot', ci_type='profile-likelihood', ci_level=None):
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
#        palette_blue = self._color_to_palette(color='b')
        self.palette = self._seaborn_palette(color='muted', cmap_len=10)

        g = sns.FacetGrid(self.df_data)
        self._plot_points(self.col_n_app, 'grtn', [self.palette[0]],
                          ax=g.ax)
        if self.cost_n_social > 0:
            try:
                self._plot_points(self.col_nup_soil_fert, 'social_cost_n',
                                  [self.palette[8]], ax=g.ax)
            except KeyError as err:
                print('{0}\nFixed social cost of N was used, so points will '
                      'not be plotted..'.format(err))
        self._modify_axes_labels(fontsize=14)
        g = self._add_ci(g, ci_type=ci_type, ci_level=ci_level)
        g = self._draw_n_cost(g, self.palette)
        g = self._modify_minmax_axes(g, x_min, x_max, y_min, y_max)
        g = self._modify_axes_pos(g)
        g = self._place_legend(g)
        g = self._modify_plot_params(g, plotsize_x=7, plotsize_y=4,
                                     labelsize=11)
        g = self._add_labels(g)
        g = self._add_title(g)
        plt.tight_layout()

        self.figure = g

    def save_plot(self, base_dir=None, fname='eonr_sns17w_pre.png', dpi=300):
        msg = ('A figure must be generated first. Please execute '
               'EONR.plot_eonr() first..\n')
        assert self.figure is not None, msg
        if base_dir is None:
            base_dir = self.base_dir
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
        fname = os.path.join(base_dir, fname)
        self.figure.savefig(fname, dpi=dpi)

    def plot_profile_likelihood(self):
        '''
        Plots the profile likelihood confidence curve for a range of alpha
        values (see Cook and Weisberg, 1990)
        '''
        df_ci = self.df_ci
#        def convert_axis(df_ci, ax, ax2):
#            ax2.set_ylim(df_ci['level'].iloc[0], df_ci['level'].iloc[-1])
#            ax2.figure.canvas.draw()
#            ax2.grid(False)

        fig, ax = plt.subplots()
#        ax2 = plt.twinx()
#        ax.callbacks.connect("ylim_changed", convert_axis(df_ci, ax, ax2))
        sns.lineplot(df_ci['wald_l'], df_ci['t_stat'], ax=ax, color='#7b7b7b')
        sns.lineplot(df_ci['wald_u'], df_ci['t_stat'], ax=ax, color='#7b7b7b')
        sns.scatterplot(df_ci['pl_l'], df_ci['t_stat'], ax=ax)
        sns.scatterplot(df_ci['pl_u'], df_ci['t_stat'], ax=ax)
        sns.scatterplot(df_ci['boot_l'], df_ci['t_stat'], ax=ax)
        sns.scatterplot(df_ci['boot_u'], df_ci['t_stat'], ax=ax)
        ax.set_xlabel('EONR')
        ax.set_ylabel('T Statistic')

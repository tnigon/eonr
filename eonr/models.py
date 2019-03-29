# -*- coding: utf-8 -*-
"""
Copyright &copy; 2019 Tyler J Nigon. All rights reserved.
"""

import numpy as np


class Models(object):
    '''
    The Models class contains algebraic functions for fitting data

    Use:
        from eonr import EONR
        from eonr import Models

        my_eonr = EONR()
        models = Models(my_eonr)

        b0 = 2.4
        b1 = 7.3
        b2 = -0.01
        x = 150
        models.quad_plateau(x, b0, b1, b2)
    '''
    def __init__(self, EONR):
        '''
        We must handle Models as a class instance so it is available anytime
        we have to pass a dynamic (but fixed from teh perspective of the model)
        variable from EONR class (e.g., R)
        '''
        self.R = EONR.R
        self.coefs_grtn = EONR.coefs_grtn
        self.cost_n_fert = EONR.cost_n_fert
        self.coefs_social = EONR.coefs_social

    def update_eonr(self, EONR):
        '''Sets/updates all EONR variables required by the Models class'''
        self.R = EONR.R
        self.coefs_grtn = EONR.coefs_grtn
        self.cost_n_fert = EONR.cost_n_fert
        self.coefs_social = EONR.coefs_social

    def exp(self, x, a, b, c):
        '''Exponential 3-param function.'''
        return a * np.exp(b * x) + c

    def poly(self, x, c, b, a=None):
        '''
        Polynomial function (up to 3 parameters)
        '''
        if a is None:
            a = 0
        return a*x**2 + b*x + c

    def quad_plateau(self, x, b0, b1, b2):
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

    def qp_theta2(self, x, b0, theta2, b2):
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
            array_temp += ((b0 - ((2*theta2*b2) - R)*x + b2*(x**2)) *
                           (x < crit_x))
            array_temp += ((b0 - ((2*theta2*b2 - R)**2) / (4*b2)) *
                           (x >= crit_x))
            return array_temp

    def combine_rtn_cost(self, x):
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
        gross_rtn = self.quad_plateau(x, b0=b0, b1=b1, b2=b2)
        # Subtractions
        fert_cost = x * self.cost_n_fert
        if self.coefs_social['lin_r2'] > self.coefs_social['exp_r2']:
            lin_b = self.coefs_social['lin_b']
            lin_mx = self.coefs_social['lin_mx']
            social_cost = self.poly(x, c=lin_b, b=lin_mx)
        else:
            exp_a = self.coefs_social['exp_a']
            exp_b = self.coefs_social['exp_b']
            exp_c = self.coefs_social['exp_c']
            social_cost = self.exp(x, a=exp_a, b=exp_b, c=exp_c)
        result = gross_rtn - fert_cost - social_cost
        return -result

    def qp_gross_theta2(self, x, b0, theta2, b2):
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

    def qp_net(self, x, b0, theta2, b2):
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

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:50:16 2018

@author: nigo0024
"""
# In[Packages and functions]
import pandas as pd
import numpy as np
import pint
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

from eonr import eonr

def replace_missing_vals(df, missing_val=['.', '#VALUE!'], cols_numeric=None):
    '''
    Finds missing data in pandas dataframe and replaces with np.nan
    '''
    if isinstance(missing_val, str):
        df.replace(missing_val, np.nan, inplace=True)
    else:
        for m_val in missing_val:
            df.replace(missing_val, np.nan, inplace=True)
    df = df.dropna()
    if cols_numeric is not None:
        df[cols_numeric] = df[cols_numeric].apply(pd.to_numeric)
    return df

# In[Load data]
pc = 'agrobot'  # 'agrobot' or 'yoga'
units = 'imperial'  # 'metric' or 'imperial'

if pc == 'agrobot':
    data_dir = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2018_eonr\data\obs'
else:
    data_dir = r'C:\Users\Tyler\Dropbox\UMN\UMN_Publications\2018_eonr\data\obs'

if units == 'metric':
    sns_fname = os.path.join(data_dir, 'sns_harvest_metric.csv')
    nue_fname = os.path.join(data_dir, 'nue_strips_harvest_metric.csv')
else:
    sns_fname = os.path.join(data_dir, 'sns_harvest_imperial.csv')
    nue_fname = os.path.join(data_dir, 'nue_strips_harvest_imperial.csv')

df_full_sns = pd.read_csv(sns_fname)
df_full_nue = pd.read_csv(nue_fname)

if units == 'metric':
    col_n_app = 'rate_n_applied_kgha'
    col_yld = 'yld_grain_dry_kgha'
    col_crop_nup = 'nup_total_kgha'
    col_nup_soil_fert = 'soil_plus_fert_n_kgha'
else:
    col_n_app = 'rate_n_applied_lbac'
    col_yld = 'yld_grain_dry_buac'
    col_crop_nup = 'nup_total_lbac'
    col_nup_soil_fert = 'soil_plus_fert_n_lbac'

col_names = ['year', 'location', 'plot', 'trt', 'rep', 'time_n',
             col_n_app, col_yld, col_crop_nup, col_nup_soil_fert]
cols_numeric = [col_n_app, col_yld, col_crop_nup, col_nup_soil_fert]
df_eonr_sns = replace_missing_vals(df_full_sns[col_names], cols_numeric=cols_numeric)
df_eonr_nue = replace_missing_vals(df_full_nue[col_names], cols_numeric=cols_numeric)

# In[Build dataframes for loc, year, and N timing]

loc = 'Gaylord'
year = 2012
df_nue12g = df_eonr_nue[(df_eonr_nue['location']==loc) & (df_eonr_nue['year']==year)]
df_temp = df_nue12g.copy()
df_nue12g_pre = df_temp[(df_temp['time_n']=='Pre')]
#df_temp_n0 = df_nue12g_pre[df_nue12g_pre[col_n_app] == 0]
#df_temp_n0.loc[:,'time_n'] = 'V5'
#df_nue12g_v5 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V5')]], ignore_index=True).reindex()
#df_temp_n0.loc[:,'time_n'] = 'V10'
#df_nue12g_v10 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V10')]], ignore_index=True).reindex()

loc = 'Stewart'
year = 2012
df_nue12s = df_eonr_nue[(df_eonr_nue['location']==loc) & (df_eonr_nue['year']==year)]
df_temp = df_nue12s.copy()
df_nue12s_pre = df_temp[(df_temp['time_n']=='Pre')]

loc = 'Janesville'
year = 2013
df_nue13j = df_eonr_nue[(df_eonr_nue['location']==loc) & (df_eonr_nue['year']==year)]
df_temp = df_nue13j.copy()
df_nue13j_pre = df_temp[(df_temp['time_n']=='Pre')]

loc = 'Willmar'
year = 2013
df_nue13wil = df_eonr_nue[(df_eonr_nue['location']==loc) & (df_eonr_nue['year']==year)]
df_temp = df_nue13wil.copy()
df_nue13wil_pre = df_temp[(df_temp['time_n']=='Pre')]
# Following is from: F:\nigo0024\Documents\GitHub\PhD_project\sns_eonr\nue_strips_troubleshooting.py
# missing_data = [3, 7, 8, 11, 12, 16]
loc_low = loc + '_low'
df_nue13wil_pre_low = df_nue13wil_pre[df_nue13wil_pre['rep'].isin([2, 4, 5, 9, 13])]
df_nue13wil_pre_low.loc[:,'location'] = loc_low
loc_high = loc + '_high'
df_nue13wil_pre_high = df_nue13wil_pre[df_nue13wil_pre['rep'].isin([1, 6, 10, 14, 15])]
df_nue13wil_pre_high.loc[:,'location'] = loc_high

loc = 'NewRichland'
year = 2014
df_nue14nr = df_eonr_nue[(df_eonr_nue['location']==loc) & (df_eonr_nue['year']==year)]
df_temp = df_nue14nr.copy()
df_nue14nr_pre = df_temp[(df_temp['time_n']=='Pre')]

loc_low = loc + '_low'
df_nue14nr_pre_low = df_nue14nr_pre[df_nue14nr_pre['rep'].isin([1, 2, 3, 4, 5, 6, 7, 8])]
df_nue14nr_pre_low.loc[:,'location'] = loc_low
loc_high = loc + '_high'
df_nue14nr_pre_high = df_nue14nr_pre[df_nue14nr_pre['rep'].isin([9, 10, 11, 12, 13, 14, 15, 16])]
df_nue14nr_pre_high.loc[:,'location'] = loc_high

loc = 'SaintCharles'
year = 2014
df_nue14sc = df_eonr_nue[(df_eonr_nue['location']==loc) & (df_eonr_nue['year']==year)]
df_temp = df_nue14sc.copy()
df_nue14sc_pre = df_temp[(df_temp['time_n']=='Pre')]

loc_low1 = loc + '_low1'  # lowest yield
df_nue14sc_pre_low1 = df_nue14sc_pre[df_nue14sc_pre['rep'].isin([5, 6, 7, 8])]
df_nue14sc_pre_low1.loc[:,'location'] = loc_low1
loc_low2 = loc + '_low2'
df_nue14sc_pre_low2 = df_nue14sc_pre[df_nue14sc_pre['rep'].isin([9, 10, 11, 12])]
df_nue14sc_pre_low2.loc[:,'location'] = loc_low2

loc_high1 = loc + '_high'
df_nue14sc_pre_high1 = df_nue14sc_pre[df_nue14sc_pre['rep'].isin([13, 14, 15, 16])]
df_nue14sc_pre_high1.loc[:,'location'] = loc_high1
loc_high2 = loc + '_high'  # highest yield
df_nue14sc_pre_high2 = df_nue14sc_pre[df_nue14sc_pre['rep'].isin([1, 2, 3, 4])]
df_nue14sc_pre_high2.loc[:,'location'] = loc_high2

loc = 'Stewart'
year = 2015
df_sns15s = df_eonr_sns[(df_eonr_sns['location']==loc) & (df_eonr_sns['year']==year)]
df_temp = df_sns15s.copy()
df_sns15s_pre = df_temp[(df_temp['time_n']=='Pre')]

# add 0 N rate (labeled "Pre" timing) to V5 and V10 dataframes
df_temp_n0 = df_sns15s_pre[df_sns15s_pre[col_n_app] == 0]
df_temp_n0.loc[:,'time_n'] = 'V5'
df_sns15s_v5 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V5')]], ignore_index=True).reindex()
df_temp_n0.loc[:,'time_n'] = 'V10'
df_sns15s_v10 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V10')]], ignore_index=True).reindex()

loc = 'Waseca'
year = 2015
df_sns15w = df_eonr_sns[(df_eonr_sns['location']==loc) & (df_eonr_sns['year']==year)]
df_temp = df_sns15w.copy()
df_sns15w_pre = df_temp[(df_temp['time_n']=='Pre')]
df_temp_n0 = df_sns15w_pre[df_sns15w_pre[col_n_app] == 0]
df_temp_n0.loc[:,'time_n'] = 'V5'
df_sns15w_v5 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V5')]], ignore_index=True).reindex()
df_temp_n0.loc[:,'time_n'] = 'V10'
df_sns15w_v10 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V10')]], ignore_index=True).reindex()

loc = 'Waseca'
year = 2016
df_sns16w = df_eonr_sns[(df_eonr_sns['location']==loc) & (df_eonr_sns['year']==year)]
df_temp = df_sns16w.copy()
df_sns16w_pre = df_temp[(df_temp['time_n']=='Pre')]
df_temp_n0 = df_sns16w_pre[df_sns16w_pre[col_n_app] == 0]
df_temp_n0.loc[:,'time_n'] = 'V5'
df_sns16w_v5 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V5')]], ignore_index=True).reindex()
df_temp_n0.loc[:,'time_n'] = 'V10'
df_sns16w_v10 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V10')]], ignore_index=True).reindex()

loc = 'Waseca'
year = 2017
df_sns17w = df_eonr_sns[(df_eonr_sns['location']==loc) & (df_eonr_sns['year']==year)]
df_temp = df_sns17w.copy()
df_sns17w_pre = df_temp[(df_temp['time_n']=='Pre')]
df_temp_n0 = df_sns17w_pre[df_sns17w_pre[col_n_app] == 0]
df_temp_n0.loc[:,'time_n'] = 'V5'
df_sns17w_v5 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V5')]], ignore_index=True).reindex()
df_temp_n0.loc[:,'time_n'] = 'V10'
df_sns17w_v10 = pd.concat([df_temp_n0, df_temp[(df_temp['time_n']=='V10')]], ignore_index=True).reindex()

# In[Set units]
cost_n_fert = 0.40  # in USD per lb
cost_n_social = 0  # in USD per lb lost
price_grain = 4.00  # in USD

def imperial_to_metric(cost_n_fert=0.40, cost_n_social=0, price_grain=4.00):
    ureg = pint.UnitRegistry()
    cost_n_fert = ureg.convert(cost_n_fert, ureg.kilogram, ureg.pound)
    cost_n_social = ureg.convert(cost_n_social, ureg.kilogram, ureg.pound)
#    price_grain = price_grain / ureg.convert(56, ureg.pound, ureg.tonne)
    price_grain = price_grain / ureg.convert(56, ureg.pound, ureg.kilogram)
    return cost_n_fert, cost_n_social, price_grain

if units == 'metric':
    cost_n_fert, cost_n_social, price_grain = imperial_to_metric(cost_n_fert,
                                                                 cost_n_social,
                                                                 price_grain)
    unit_currency = '$'
    unit_fert = 'kg'
    unit_grain = 'kg'
    unit_area = 'ha'
else:
    unit_currency = '$'
    unit_fert = 'lbs'
    unit_grain = 'bu'
    unit_area = 'ac'

# In[Calculate EONR for all years]
def plot_and_save(my_eonr, fname, x_min=-5,
                  x_max=365, y_min=-50, y_max=1100, dpi=300):
    my_eonr.plot_eonr(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    if my_eonr.base_dir is not None:
        my_eonr.save_plot(fname=fname, dpi=dpi)

def calc_all_siteyears(my_eonr, print_plot=False, y_min=-50,
                       y_max=1100):
    # Waseca 2017
    my_eonr.calculate_eonr(df_sns17w_pre)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns17w_pre.png', y_min=y_min, y_max=y_max)
    my_eonr.calculate_eonr(df_sns17w_v5)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns17w_v5.png', y_min=y_min, y_max=y_max)
    my_eonr.calculate_eonr(df_sns17w_v10)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns17w_v10.png', y_min=y_min, y_max=y_max)
    # Waseca 2016
    my_eonr.calculate_eonr(df_sns16w_pre)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns16w_pre.png', y_min=y_min, y_max=y_max)
    my_eonr.calculate_eonr(df_sns16w_v5)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns16w_v5.png', y_min=y_min, y_max=y_max)
    my_eonr.calculate_eonr(df_sns16w_v10)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns16w_v10.png', y_min=y_min, y_max=y_max)
    # Waseca 2015
    my_eonr.calculate_eonr(df_sns15w_pre)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns15w_pre.png', y_min=y_min, y_max=y_max)
    my_eonr.calculate_eonr(df_sns15w_v5)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns15w_v5.png', y_min=y_min, y_max=y_max)
    my_eonr.calculate_eonr(df_sns15w_v10)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns15w_v10.png', y_min=y_min, y_max=y_max)
    # Stewart 2015
    my_eonr.calculate_eonr(df_sns15s_pre)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns15s_pre.png', y_min=y_min, y_max=y_max)
    my_eonr.calculate_eonr(df_sns15s_v5)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns15s_v5.png', y_min=y_min, y_max=y_max)
    my_eonr.calculate_eonr(df_sns15s_v10)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_sns15s_v10.png', y_min=y_min, y_max=y_max)
    # Gaylord 2012
    my_eonr.calculate_eonr(df_nue12g_pre)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue12g_pre.png', y_min=y_min, y_max=y_max)
    # Stewart 2012
    my_eonr.calculate_eonr(df_nue12s_pre)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue12s_pre.png', y_min=y_min, y_max=y_max)
    # Janesville 2013
    my_eonr.calculate_eonr(df_nue13j_pre)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue13j_pre.png', y_min=y_min, y_max=y_max)
    # Willmar 2013 - There is a lot of yield variability, so plots were divided into low yielding and high yielding
    my_eonr.calculate_eonr(df_nue13wil_pre)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue13wil_pre.png', y_min=y_min, y_max=y_max)
    # Willmar 2013 (Reps 2, 4, 5, 9, & 13)
    my_eonr.calculate_eonr(df_nue13wil_pre_low)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue13wil_pre_low.png', y_min=y_min, y_max=y_max)
    # Willmar 2013 (Reps 1, 6, 10, 14, & 15)
    my_eonr.calculate_eonr(df_nue13wil_pre_high)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue13wil_pre_high.png', y_min=y_min, y_max=y_max)
    # New Richland 2014
    my_eonr.calculate_eonr(df_nue14nr_pre)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue14nr_pre.png', y_min=y_min, y_max=y_max)
    # New Richland 2014 (Reps 1-8)
    my_eonr.calculate_eonr(df_nue14nr_pre_low)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue14nr_pre_low.png', y_min=y_min, y_max=y_max)
    # New Richland 2014 (Reps 9-16)
    my_eonr.calculate_eonr(df_nue14nr_pre_high)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue14nr_pre_high.png', y_min=y_min, y_max=y_max)
#    # Saint Charles 2014
#    my_eonr.calculate_eonr(df_nue14sc_pre)
#    if print_plot is True:
#        plot_and_save(my_eonr, fname='eonr_nue14sc_pre.png', y_min=y_min, y_max=y_max)
#    # Saint Charles 2014 (low1 (reps 5-8))
#    my_eonr.calculate_eonr(df_nue14sc_pre_low1)
#    if print_plot is True:
#        plot_and_save(my_eonr, fname='eonr_nue14sc_pre_low1.png', y_min=y_min, y_max=y_max)
#    # Saint Charles 2014 (low2 (reps 9-12))
#    my_eonr.calculate_eonr(df_nue14sc_pre_low2)
#    if print_plot is True:
#        plot_and_save(my_eonr, fname='eonr_nue14sc_pre_low2.png', y_min=y_min, y_max=y_max)
#    # Saint Charles 2014 (high1 (reps 13-16))
#    my_eonr.calculate_eonr(df_nue14sc_pre_high1)
#    if print_plot is True:
#        plot_and_save(my_eonr, fname='eonr_nue14sc_pre_high1.png', y_min=y_min, y_max=y_max)
    # Saint Charles 2014 (high2 (reps 1-4))
    my_eonr.calculate_eonr(df_nue14sc_pre_high2)
    if print_plot is True:
        plot_and_save(my_eonr, fname='eonr_nue14sc_pre_high2.png', y_min=y_min, y_max=y_max)
    return my_eonr

# In[Run EONR function]
base_dir = r'G:\SOIL\GIS\SNS\eonr\2019-02-15\imperial'
#base_dir = r'C:\Users\Tyler\eonr\2019-02-10'
my_eonr = eonr(cost_n_fert=cost_n_fert,
               cost_n_social=cost_n_social,
               price_grain=price_grain,
               col_n_app=col_n_app,
               col_yld=col_yld,
               col_crop_nup=col_crop_nup,
               col_nup_soil_fert=col_nup_soil_fert,
               unit_currency=unit_currency,
               unit_grain=unit_grain,
               unit_fert=unit_fert,
               unit_area=unit_area,
               model='quad_plateau',
               ci_level=0.9,
               base_dir=base_dir,
               base_zero=True,
               print_out=False)

# In[Run traditional]
social = False
print_plot = True
y_min = -50
y_max = 550

cost_n_fert_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
#cost_n_fert_list = [0.3]
for cost_n_fert in cost_n_fert_list:
    cost_n_social = 0
    price_grain = 4.00
    cost_n_fert = 0.5
    if units == 'metric':
        y_min = -100
        y_max = 1600
        cost_n_fert, cost_n_social, price_grain = imperial_to_metric(
                cost_n_fert=cost_n_fert, cost_n_social=cost_n_social,
                price_grain=price_grain)
    my_eonr.set_ratio(cost_n_fert=cost_n_fert,
                      cost_n_social=cost_n_social,
                      price_grain=price_grain)
    my_eonr = calc_all_siteyears(my_eonr, print_plot=print_plot, y_min=y_min,
                                 y_max=y_max)
    plt.close("all")

my_eonr.df_results.to_csv(os.path.join(os.path.split(my_eonr.base_dir)[0], 'trad_results.csv'))
my_eonr.df_ci.to_csv(os.path.join(os.path.split(my_eonr.base_dir)[0], 'trad_ci.csv'))

# In[Run social]
social = True
print_plot = True
y_min = -200
y_max = 1100

cost_n_social_list = [0.01, 0.1, 0.25, 0.5, 1, 2, 5]

for cost_n_social in cost_n_social_list:
    cost_n_fert = 0.4
    price_grain = 4.00
    if units == 'metric':
        y_min = -400
        y_max = 2000
        cost_n_fert, cost_n_social, price_grain = imperial_to_metric(
                cost_n_fert=cost_n_fert, cost_n_social=cost_n_social,
                price_grain=price_grain)
    my_eonr.set_ratio(cost_n_fert=cost_n_fert,
                      cost_n_social=cost_n_social,
                      price_grain=price_grain)
    my_eonr = calc_all_siteyears(my_eonr, print_plot=print_plot, y_min=y_min,
                                 y_max=y_max)
    plt.close("all")

my_eonr.df_results.to_csv(os.path.join(os.path.split(my_eonr.base_dir)[0], 'social_results.csv'))
my_eonr.df_ci.to_csv(os.path.join(os.path.split(my_eonr.base_dir)[0], 'social_ci.csv'))

# In[Plot all]
df_results = my_eonr.df_results

fname = r'G:\SOIL\GIS\SNS\eonr\eonr_social1.csv'
df_results.to_csv(fname, index=False)
df_results_dif1 = pd.read_csv(fname)

fname2 = r'G:\SOIL\GIS\SNS\eonr\trad_v3_final\eonr_trad3.csv'
df_results_dif2 = pd.read_csv(fname2) # traditional

fig1, ax1 = plt.subplots()
ax1 = sns.scatterplot(x='eonr', y='mrtn', hue='time_n', size='r2_adj', data=df_results)

# In[Plot functions]
def _add_title(title_text, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="15%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_facecolor('#7b7b7b')
    at = AnchoredText(title_text, loc=10, frameon=False,
                      prop=dict(backgroundcolor='#7b7b7b',
                                size=11.5, color='white',
                                fontweight='bold'))
    cax.add_artist(at)

# In[Plot EONR difference in subplots]
df = df_results_dif1.copy()
fname_out = r'G:\SOIL\GIS\SNS\eonr\eonr_social1_priceratio.png'

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
xlabel = 'Price Ratio'
ylabel = ('EONR Difference (lb ac$^{-1}$)')

df_ax1 = df[(df['location'] == 'Waseca') & (df['year'] == 2015)]
df_ax2 = df[(df['location'] == 'Waseca') & (df['year'] == 2016)]
df_ax3 = df[(df['location'] == 'Waseca') & (df['year'] == 2017)]
df_ax4 = df[(df['location'] == 'Stewart') & (df['year'] == 2015)]

sns.pointplot(x='price_ratio', y='dif_05', hue='time_n', data=df_ax1, ax=ax1, palette='muted')
sns.pointplot(x='price_ratio', y='dif_05', hue='time_n', data=df_ax2, ax=ax2, palette='muted')
sns.pointplot(x='price_ratio', y='dif_05', hue='time_n', data=df_ax3, ax=ax3, palette='muted')
sns.pointplot(x='price_ratio', y='dif_05', hue='time_n', data=df_ax4, ax=ax4, palette='muted')

plt.setp(ax1.collections, sizes=[30])
plt.setp(ax2.collections, sizes=[30])
plt.setp(ax3.collections, sizes=[30])
plt.setp(ax4.collections, sizes=[30])

lines_list = []
lines_list.append(ax1.lines)
lines_list.append(ax2.lines)
lines_list.append(ax3.lines)
lines_list.append(ax4.lines)

for lines in lines_list:
    plt.setp(lines[0:8], linewidth=1.5, linestyle='-')
    plt.setp(lines[8:16], linewidth=1.5, linestyle='--')
    plt.setp(lines[16:24], linewidth=1.5, linestyle='-.')

_add_title('Location = Waseca | Year = 2015', ax1)
_add_title('Location = Waseca | Year = 2016', ax2)
_add_title('Location = Waseca | Year = 2017', ax3)
_add_title('Location = Stewart | Year = 2015', ax4)

ax1.set_xlabel('')
ax2.set_xlabel('')
ax3.set_xlabel(xlabel, weight='bold')
ax4.set_xlabel(xlabel, weight='bold')
ax3.set_xticklabels(ax3.get_xmajorticklabels(), rotation=60)
ax4.set_xticklabels(ax4.get_xmajorticklabels(), rotation=60)

ax1.set_ylabel(ylabel, weight='bold')
ax2.set_ylabel('')
ax3.set_ylabel(ylabel, weight='bold')
ax4.set_ylabel('')

leg1 = ax1.legend()
leg2 = ax2.legend()
leg3 = ax3.legend()

h, labels = ax4.get_legend_handles_labels()
leg1.remove()
leg2.remove()
leg3.remove()
incr = int(len(lines_list[3]) / 3)
leg4 = ax4.legend((lines_list[3][0], lines_list[3][incr], lines_list[3][incr*2]),
                  labels, loc='lower center')
for text in leg4.get_texts():
    text.set_color('#555555')

f.set_size_inches(8,8)
f.tight_layout()
f.tight_layout()  # not sure why, but this has to be here twice..

f.savefig(fname_out, bbox_inches="tight")

# In[Clean digitized data]
import pandas as pd
import numpy as np

def interpolate_daily(fname):
    '''
    Takes csv data from a digitized plot (https://apps.automeris.io/wpd/),
    takes mean of duplicate values, and interpolates to daily values. Returns a
    daily dataframe.
    '''
    df = pd.read_csv(fname, header=None, names=['date', 'usd_lb_N'],
                     parse_dates={'dt' : ['date']})
    df = df.set_index('dt')
    df = df.sort_index()
    df_dly = df.resample('D').mean().interpolate('linear')
    df_monthly = df_dly[df_dly.index.day == 1]
    return df_dly, df_monthly

fname_aa = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2018_eonr\appendix\dtn_aa_price_digitize.csv'
_, df_aa = interpolate_daily(fname_aa)
df_aa.columns = ['aa_usd_lb']

fname_urea = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2018_eonr\appendix\dtn_urea_price_digitize.csv'
_, df_urea = interpolate_daily(fname_urea)
df_urea.columns = ['urea_usd_lb']

fname_uan28 = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2018_eonr\appendix\dtn_uan28_price_digitize.csv'
_, df_uan28 = interpolate_daily(fname_uan28)
df_uan28.columns = ['uan28_usd_lb']

fname_uan32 = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2018_eonr\appendix\dtn_uan32_price_digitize.csv'
_, df_uan32 = interpolate_daily(fname_uan32)
df_uan32.columns = ['uan32_usd_lb']

# In[Prep cost data]

# Monthly grain
fname_grain = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2018_eonr\data\nass\MN_corn_grain_price_all.csv'
df_grain = pd.read_csv(fname_grain)
df_grain.index = (pd.to_datetime(df_grain['year'].astype(str) + df_grain['month'].astype(str), format='%Y%m'))
df_grain = df_grain[df_grain.index.year > 1995].sort_index()
df_grain = df_grain[['price_usd_bu']]

# Annual fertilizer (1996-2014)
fname_prices = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2018_eonr\data\nass\MN_N_corn_prices.csv'
df_prices = pd.read_csv(fname_prices)
df_prices.index = (pd.to_datetime(df_prices['year'].astype(str) + df_prices['month'].astype(str), format='%Y%m'))
df_prices = df_prices.sort_index()
df_prices = df_prices[['price_aa', 'price_nsol', 'price_urea', 'price_nh4no3']]

df_merge = df_grain.join(df_prices, how='left', rsuffix='fert')
df_merge = df_merge.join(df_aa, how='left')
df_merge = df_merge.join(df_uan32, how='left')
df_merge = df_merge.join(df_urea, how='left')

df_merge['price2_aa'] = np.NaN
df_merge['price2_nsol'] = np.NaN
df_merge['price2_urea'] = np.NaN
df_merge['price2_nh4no3'] = np.NaN

df_merge['ratio_aa'] = np.NaN
df_merge['ratio_nsol'] = np.NaN
df_merge['ratio_urea'] = np.NaN
df_merge['ratio_nh4no3'] = np.NaN

df_merge1 = df_merge[df_merge.index.year < 2010]
df_merge2 = df_merge[df_merge.index.year >= 2010]

df_merge.loc[:, 'price2_aa'] = pd.concat([df_merge1['price_aa'], df_merge2['aa_usd_lb']])
df_merge['ratio_aa'] = df_merge['price2_aa'] / df_merge['price_usd_bu']

df_merge.loc[:, 'price2_nsol'] = pd.concat([df_merge1['price_nsol'], df_merge2['uan32_usd_lb']])
df_merge['ratio_nsol'] = df_merge['price2_nsol'] / df_merge['price_usd_bu']

df_merge.loc[:, 'price2_urea'] = pd.concat([df_merge1['price_urea'], df_merge2['urea_usd_lb']])
df_merge['ratio_urea'] = df_merge['price2_urea'] / df_merge['price_usd_bu']

df_merge.loc[:, 'ratio_nh4no3'] = df_merge1['price_nh4no3'] / df_merge1['price_usd_bu']

# In[Plot Historical price ratio]
import pandas as pd
import seaborn as sns
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use('ggplot')
pal = sns.color_palette('muted', 8)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

sns.lineplot(x=df_merge.index, y='price_usd_bu', data=df_merge, ax=ax1)
sns.lineplot(data=df_merge[['price2_aa', 'price2_nsol', 'price2_urea', 'price2_nh4no3']],
             palette=pal[0:4], ax=ax2)
sns.lineplot(data=df_merge[['ratio_aa', 'ratio_nsol', 'ratio_urea', 'ratio_nh4no3']],
             palette=pal[0:4], ax=ax3)

# Legend
custom_lines = [Line2D([0], [0], color=pal[7], lw=2, linestyle='--')]
ax1.legend(custom_lines,
           ['Maize Grain Price'],
           loc='upper left',
           frameon=True,
           framealpha=0.75,
           facecolor='white',
           fancybox=True,
           borderpad=0.5,
           edgecolor=(0.5, 0.5, 0.5),
           prop={
                 'weight': 'bold',
                 'size': 9
                 })

handles, labels1 = ax2.get_legend_handles_labels()
labels = ['Anhydrous Ammonia', 'UAN Solution', 'Urea', 'Ammonium Nitrate']
order = [0, 2, 3, 1]
handles = [handles[idx] for idx in order]
labels = [labels[idx] for idx in order]
ax2.get_legend().remove()
ax3.get_legend().remove()

plt.figlegend(handles,
              labels,
              loc='lower center',
              ncol=4,
              frameon=True,
              framealpha=0.75,
              facecolor='white',
              fancybox=True,
              borderpad=0.5,
              edgecolor=(0.5, 0.5, 0.5),
              prop={
                      'weight': 'bold',
                      'size': 9
                      })

# clean up Plot
fig.suptitle('Nitrogen Fertilizer:Grain Price Ratios (1996 - 2018)', weight='bold')
ax3.set_xlabel('Year', weight='bold')
ax1.set_ylabel('Price per bu ($)', weight='bold')
ax2.set_ylabel('Cost per lb N ($)', weight='bold')
ax3.set_ylabel('Price Ratio', weight='bold')

ax1_ticks = ax1.get_yticks().tolist()
ax1_ticks_new = []
for item in ax1_ticks:
    ax1_ticks_new.append('${0:.2f}'.format(float(item)))
ax1.set_yticklabels(ax1_ticks_new)

ax2_ticks = ax2.get_yticks().tolist()
ax2_ticks_new = []
for item in ax2_ticks:
    ax2_ticks_new.append('${0:.2f}'.format(float(item)))
ax2.set_yticklabels(ax2_ticks_new)

ax3_ticks = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
ax3_tick_labels = []
for item in ax3_ticks:
    ax3_tick_labels.append(str(item))
ax3.set_yticks(ax3_ticks)

plt.rcParams['font.weight'] = 'bold'
plt.tight_layout()
plt.subplots_adjust(bottom=0.14, top=0.95)

# In[Plot Grain price, fertilizer cost, and price ratio]
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
pal = sns.color_palette('muted', 8)
#sns.palplot(pal)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

sns.lineplot(x=df_merge.index, y='price_usd_bu', data=df_merge, ax=ax1)
sns.lineplot(data=df_merge[['price2_aa', 'price2_nsol', 'price2_urea']],
             palette=pal[0:3], ax=ax2)
sns.lineplot(data=df_merge[['ratio_aa', 'ratio_nsol', 'ratio_urea']],
             palette=pal[0:3], ax=ax3)

fig.suptitle('Nitrogen Fertilizer:Grain Price Ratios (1996 - 2018)',
          fontweight='bold', fontsize=15)
ax1.set_ylabel('Price per bu ($)', fontweight='bold')
ax2.set_ylabel('Cost per lb N ($)', fontweight='bold')
ax3.set_ylabel('Price Ratio', fontweight='bold')

ax3.set_xlabel('Date', fontweight='bold')




# In[Corn usage by segment]
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
pal = sns.color_palette('muted', 8)

fname_cornuse = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2018_eonr\data\nass\feed_grains_use_organized.csv'
df_cornuse = pd.read_csv(fname_cornuse)

df_cornuse_tot1 = df_cornuse.iloc[1:3]
df_cornuse_tot2 = df_cornuse.iloc[3:6]
df_cornuse_dom = df_cornuse.iloc[6:]

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

ax1 = sns.catplot(x="million_bu", y='level1', data=df_cornuse_tot1, kind="bar", palette="muted", ax=ax1)
ax2 = sns.catplot(x="million_bu", y='level2', data=df_cornuse_tot2, kind="bar", palette="muted", ax=ax2)
ax3 = sns.catplot(x="million_bu", y='level3', data=df_cornuse_dom, kind="bar", palette="muted", ax=ax3)

df_cornuse_dom = df_cornuse.iloc[5:]

ct = pd.crosstab(df_cornuse.million_bu, df_cornuse.level2)
ct.plot.bar(stacked=True)


# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:31:32 2017

@author: nigo0024
"""
# In[Packages and functions]
import pandas as pd
import numpy as np


def replace_missing_vals(df, missing_val='.'):
    '''
    Finds missing data in pandas dataframe and replaces with np.nan
    '''
    for row in df.index:
        for col in df.columns:
            if df.at[row, col] == missing_val:
                df.at[row, col] = np.nan
    return df

sns_fname = r'G:\SOIL\GIS\Master Data Files\SNS_consistent_heading.xlsx'
sns_xlsx = pd.ExcelFile(sns_fname)
print(sns_xlsx.sheet_names)
df_agron = pd.read_excel(sns_fname, 'Agronomic')
df_repeat = pd.read_excel(sns_fname, 'Repeated')

# Calculate EONR
cost_N = 0.30  # in USD per lb
price_grain15 = 3.00  # in USD
x_min=-5
x_max=365
y_min=0
y_max=800

# ========================== Waseca 2015 ===================================
df_2015W = df_agron[(df_agron['Location']=='Waseca') & (df_agron['Year']==2015)]
df_2015W_pre = df_2015W[(df_2015W['Time']=='Pre')]
df_2015W_V5 = df_2015W[(df_2015W['Time']=='V5')]
df_2015W_V10 = df_2015W[(df_2015W['Time']=='V10')]

df_repeat = replace_missing_vals(df_repeat, missing_val='.')
df_repeat_2015W = df_repeat[(df_repeat['Location']=='Waseca') & (df_repeat['Year']==2015)]
df_repeat_2016W = df_repeat[(df_repeat['Location']=='Waseca') & (df_repeat['Year']==2016)]
print(df_repeat_2015W.columns)
df_repeat_2015W_pre = df_repeat_2015W[(df_repeat_2015W['Time']=='Pre')]
df_repeat_2015W_v5 = df_repeat_2015W[(df_repeat_2015W['Time']=='V5')]
df_repeat_2015W_v10 = df_repeat_2015W[(df_repeat_2015W['Time']=='V10')]

# ======================================================================
#import EONR
dpi = 300
eonr_W15_pre = EONR(df_2015W_pre, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W15_pre.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
fname_out = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\NCEISFC_2017\eonr_W15_pre_10.png'
eonr_W15_pre.figure.savefig(fname_out, dpi=dpi)

eonr_W15_v5 = EONR(df_2015W_V5, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W15_v5.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
fname_out = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\NCEISFC_2017\eonr_W15_v5_10.png'
eonr_W15_v5.figure.savefig(fname_out, dpi=dpi)

eonr_W15_v10 = EONR(df_2015W_V10, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W15_v10.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
fname_out = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\NCEISFC_2017\eonr_W15_v10_10.png'
eonr_W15_v10.figure.savefig(fname_out, dpi=dpi)


# In[2017 NCEISFC]

sns_fname = r'G:\SOIL\GIS\Master Data Files\SNS_consistent_heading.xlsx'
sns_xlsx = pd.ExcelFile(sns_fname)
print(sns_xlsx.sheet_names)
df_agron = pd.read_excel(sns_fname, 'Agronomic')
df_agron = replace_missing_vals(df_agron, missing_val='.')
df_repeat = pd.read_excel(sns_fname, 'Repeated')
df_repeat = replace_missing_vals(df_repeat, missing_val='.')
timing_N = df_agron[(df_agron['Location']=='Waseca') & (df_agron['Year']==2015)].Time  # timing of N application
rate_N = df_agron[(df_agron['Location']=='Waseca') & (df_agron['Year']==2015)].Nrate  # in lbs per acre
grain_yield = df_agron[(df_agron['Location']=='Waseca') & (df_agron['Year']==2015)].YieldOUT  # in bushels per acre

# In[Querying pandas dataframe examples]
test = df_agron[df_agron['Location']=='Waseca']
test = df_agron[(df_agron['Location']=='Waseca') & (df_agron['Year']==2015)]
test = df_agron[(df_agron['Location']=='Waseca') & (df_agron['Year']==2015) & (df_agron['TRT']==1)]

 Get grouping means:
df_agron[(df_agron['Location']=='Waseca') & (df_agron['Year']==2015)].groupby('TRT').YieldOUT.mean()

# In[Waseca 2016]
df_2016W = df_agron[(df_agron['Location']=='Waseca') & (df_agron['Year']==2016)]
df_2016W_pre = df_2016W[(df_2016W['Time']=='Pre')]
df_2016W_V5 = df_2016W[(df_2016W['Time']=='V5')]
df_2016W_V10 = df_2016W[(df_2016W['Time']=='V10')]

eonr_W16_pre = EONR(df_2016W_pre, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W16_pre.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
fname_out = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\NCEISFC_2017\eonr_W16_pre_10.png'
eonr_W16_pre.figure.savefig(fname_out, dpi=dpi)

eonr_W16_v5 = EONR(df_2016W_V5, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W16_v5.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
fname_out = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\NCEISFC_2017\eonr_W16_v5_10.png'
eonr_W16_v5.figure.savefig(fname_out, dpi=dpi)

eonr_W16_v10 = EONR(df_2016W_V10, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W16_v10.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
fname_out = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\NCEISFC_2017\eonr_W16_v10_10.png'
eonr_W16_v10.figure.savefig(fname_out, dpi=dpi)

# In[Waseca 2015 by rep]
def EONR_byrep(df, cost_N, price_grain, Napplied_str='Napplied',
               yield_str='YieldOUT', show_plot=False):
    '''Calculates the EONR for each replication'''
    array_reps = df.Rep.unique()
    eonr_dict = {}
    for rep in array_reps.tolist():
        df_rep = df[(df['Rep'] == rep)]
        dict_key = 'Key{0}'.format(rep)
        eonr_dict[dict_key] = EONR(df_rep,
                                   cost_N=cost_N,
                                   price_grain=price_grain,
                                   Napplied_str='Napplied',
                                   yield_str='YieldOUT',
                                   show_plot=False)
    return eonr_dict

cost_N = 0.50  # in USD per lb
price_grain15 = 3.00  # in USD

df_2015W_pre = df_2015W[(df_2015W['Time']=='Pre')]
df_2015W_V5 = df_2015W[(df_2015W['Time']=='V5')]
df_2015W_V10 = df_2015W[(df_2015W['Time']=='V10')]

eonr_rep_W15 = EONR_byrep(df_2015W_pre, cost_N, price_grain15)


df_2015W_pre1 = df_2015W_pre[(df_2015W_pre['Rep'] == 1)]
df_2015W_pre2 = df_2015W_pre[(df_2015W_pre['Rep'] == 2)]
df_2015W_pre3 = df_2015W_pre[(df_2015W_pre['Rep'] == 3)]
df_2015W_pre4 = df_2015W_pre[(df_2015W_pre['Rep'] == 4)]

# In[import EONR]
eonr_W15_pre1 = EONR(df_2015W_pre1, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W15_pre2 = EONR(df_2015W_pre2, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W15_pre3 = EONR(df_2015W_pre3, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W15_pre4 = EONR(df_2015W_pre4, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)

eonr_W15_pre1.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
eonr_W15_pre2.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
eonr_W15_pre3.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
eonr_W15_pre4.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

eonr_W15_pre.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
fname_out = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\NCEISFC_2017\eonr_W15_pre.png'
eonr_W15_pre.figure.savefig(fname_out, dpi=150)

eonr_W15_v5 = EONR(df_2015W_V5, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W15_v5.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
fname_out = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\NCEISFC_2017\eonr_W15_v5.png'
eonr_W15_v5.figure.savefig(fname_out, dpi=150)

eonr_W15_v10 = EONR(df_2015W_V10, cost_N=cost_N, price_grain=price_grain15,
                Napplied_str='Napplied', yield_str='YieldOUT', show_plot=False)
eonr_W15_v10.Plot_EONR(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
fname_out = r'F:\nigo0024\Dropbox\UMN\UMN_Publications\NCEISFC_2017\eonr_W15_v10.png'
eonr_W15_v10.figure.savefig(fname_out, dpi=150)


# In[Plot EONR and MRTN next to each other]
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# get data into dataframe
eonr_array = np.transpose(np.array(
        [[2015, 2015, 2015, 2016, 2016, 2016],
         ['Pre', 'V5', 'V10', 'Pre', 'V5', 'V10'],
         [eonr_W15_pre.eonr, eonr_W15_v5.eonr, eonr_W15_v10.eonr,
          eonr_W16_pre.eonr, eonr_W16_v5.eonr, eonr_W16_v10.eonr],
         [eonr_W15_pre.mrtn, eonr_W15_v5.mrtn, eonr_W15_v10.mrtn,
          eonr_W16_pre.mrtn, eonr_W16_v5.mrtn, eonr_W16_v10.mrtn]]))
df_eonr = pd.DataFrame(data=eonr_array,
                         columns=np.array(['Year', 'Timing', 'EONR', 'MRTN']))
df_eonr['Year'] = pd.to_numeric(df_eonr['Year'])
df_eonr['EONR'] = pd.to_numeric(df_eonr['EONR'])
df_eonr['MRTN'] = pd.to_numeric(df_eonr['MRTN'])

rmse_df_height = rmse_df[rmse_df.Measurement == 'Plant height (cm)']
rmse_df_biomass = rmse_df[rmse_df.Measurement == 'Biomass (g)']
rmse_df_lai = rmse_df[rmse_df.Measurement == 'LAI']

sns.set(style="white", context="talk")
plt.rcParams['font.weight'] = 'bold'
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharex=True)

sns.factorplot(x='Year', y='EONR', hue='Timing', data=df_eonr,
                   size=6, kind="bar", palette="muted", ax=ax1)

sns.factorplot(x='Year', y='MRTN', hue='Timing', data=df_eonr,
                   size=6, kind="bar", palette="muted", ax=ax2)

sns.barplot(rmse_df_height['stage'], rmse_df_height['rmse'], palette="Blues", ax=ax1)
ax1.set_xlabel("Plant Height", fontweight='bold')
ax1.set_ylabel("RMSE (cm)", fontweight='bold')

count = 0
for p in ax1.patches:
    val = int(rmse_df_height['n'][count])
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height + 1.1,
            'n={:1.0f}'.format(val),
            ha="center",
            fontsize=12,
            fontweight='bold',
            rotation=45)
    count += 1

sns.barplot(rmse_df_biomass['stage'], rmse_df_biomass['rmse'], palette="Blues", ax=ax2)
ax2.set_xlabel("Above-ground Biomass", fontweight='bold')
ax2.set_ylabel("RMSE (g)", fontweight='bold')
for p in ax2.patches:
    val = int(rmse_df_biomass['n'][count])
    height = p.get_height()
    ax2.text(p.get_x()+p.get_width()/2.,
            height + 0.55,
            'n={:1.0f}'.format(val),
            ha="center",
            fontsize=12,
            fontweight='bold',
            rotation=45)
    count += 1

sns.barplot(x='stage', y= 'rmse', data=rmse_df_lai, palette="Blues", ax=ax3, label='stage')
ax3.set_xlabel("Leaf Area Index", fontweight='bold')
ax3.set_ylabel("RMSE", fontweight='bold')
for p in ax3.patches:
    val = int(rmse_df_lai['n'][count])
    height = p.get_height()
    ax3.text(p.get_x()+p.get_width()/2.,
            height + 0.032,
            'n={:1.0f}'.format(val),
            ha="center",
            fontsize=12,
            fontweight='bold',
            rotation=45)
    count += 1

# Finalize the plot
sns.despine(bottom=True)
#ax3.legend(ncol=2, loc="upper right", frameon=True)
#ax3.set(xlim=(0, 24), ylabel="",
#       xlabel="test")
plt.tight_layout(h_pad=3)

# In[Plot grain yield with confidence intervals]
def cmap_to_palette(cmap):
    '''Converts matplotlib cmap to seaborn color palette '''
    palette = []
    for row in cmap.colors:
        palette.append(tuple(row))
    return palette

def plot_yield(df, loc_str='Waseca 2015',
               means_sep = ['C', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A']):
    import seaborn as sns
    import numpy

    sns.set(style="whitegrid")
    plt.rcParams['font.weight'] = 'bold'
    cmap1 = plt.get_cmap('viridis', 3)
    palette1 = cmap_to_palette(cmap1)
    g = sns.factorplot(x="Napplied", y="YieldOUT", hue="Time", data=df,
                       palette=palette1, size=6, kind='bar', ci=95,
                       legend=False)
    ax = g.ax
    ax.set_xlabel("Nitrogen Rate (lbs per acre)",
                  fontweight='bold',
                  fontsize=14)
    ax.set_ylabel("Grain Yield (bu per acre)",
                  fontweight='bold',
                  fontsize=14)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(loc='right', bbox_to_anchor=(1.2, 0.5), fontsize=11,
              title='N Timing')
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    g.fig.subplots_adjust(top=0.92, bottom=0.12, left=0.11, right=0.86)
    g.fig.set_size_inches(6, 4)
    plt.title('Grain Yield - {0}'.format(loc_str),
              fontweight='bold', fontsize=15)
    height_max = 0
    for p in ax.patches:
        height = p.get_height()
        if height > height_max:
            height_max = height
    count = 0
    list_x = []
    for idx, p in enumerate(ax.patches):
        list_x.append(p.get_x())
    list_x_idx = [b[0] for b in sorted(enumerate(list_x), key=lambda i:i[1])]
    list_patch = ax.patches
    list_x_idx = numpy.array(list_x_idx)
    list_patch = numpy.array(list_patch)
    sorted_patch = list_patch[list_x_idx]
    for idx, p in enumerate(sorted_patch):
        if ((idx + 1) % 3) != 2:
            continue
        else:
            if count < len(means_sep):
                ax.text(p.get_x()+p.get_width()/2.,
                        height_max + (height_max * 0.06),
                        '{0}'.format(means_sep[count]),
                        ha="center",
                        fontsize=12,
                        fontweight='bold')
                count += 1

# get means_sep from SAS output
means_sep_W15 = letters = ['E', 'D', 'C', 'B', 'B', 'AB', 'AB', 'A', 'AB']
means_sep_W16 = ['E', 'D', 'C', 'B', 'A', 'A', 'A', 'A', 'A']
plot_yield(df_2015W, loc_str='Waseca 2015', means_sep=means_sep_W15)
plot_yield(df_2016W, loc_str='Waseca 2016', means_sep=means_sep_W16)

def plot_N_uptake(df, timing='V5', loc_str='Waseca 2015', y_col='V5_NUP_lbac',
               means_sep = ['C', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A']):
    import seaborn as sns

    sns.set(style="whitegrid")
    plt.rcParams['font.weight'] = 'bold'
    cmap1 = plt.get_cmap('viridis', 3)
    palette1 = cmap_to_palette(cmap1)
    g = sns.factorplot(x="Napplied", y=y_col, data=df,
                       size=6, kind='point', ci=95,
                       legend=False)
    ax = g.ax
    ax.set_xlabel("Nitrogen Rate (lbs per acre)",
                  fontweight='bold',
                  fontsize=14)
    ax.set_ylabel("Nitrogen Uptake (lbs N per acre)",
                  fontweight='bold',
                  fontsize=14)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(loc='right', bbox_to_anchor=(1.2, 0.5), fontsize=11,
              title='N Timing')
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    g.fig.subplots_adjust(top=0.92, bottom=0.12, left=0.11, right=0.86)
    g.fig.set_size_inches(6, 4)
    plt.title('{0} Nitrogen Uptake - {1}'.format(timing, loc_str),
              fontweight='bold', fontsize=15)
    count = 0
    height_max = g.ax.axes.get_ylim()[0]
    array_rates = df.Napplied.unique()
    array_rates.sort()
    for rate in array_rates:
        ax.text(count,
                height_max + (height_max * 0.06),
                '{0}'.format(means_sep[count]),
                ha="center",
                fontsize=12,
                fontweight='bold')
        count += 1
# get means_sep from SAS output
means_sep_W15_V5NUP = letters = ['B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
plot_N_uptake(df_repeat_2015W, timing='V5', loc_str='Waseca 2015',
              y_col='V5_NUP_lbac', means_sep=means_sep_W15_V5NUP)

means_sep_W15_V10NUP = letters = ['E', 'DE', 'DE', 'CD', 'BC', 'ABC', 'AB', 'A', 'BC']
plot_N_uptake(df_repeat_2015W, timing='V10', loc_str='Waseca 2015',
              y_col='V10_NUP_lbac', means_sep=means_sep_W15_V10NUP)

means_sep_W16_V5NUP = letters = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
plot_N_uptake(df_repeat_2016W, timing='V5', loc_str='Waseca 2016',
              y_col='V5_NUP_lbac', means_sep=means_sep_W16_V5NUP)

means_sep_W16_V10NUP = letters = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
plot_N_uptake(df_repeat_2016W, timing='V10', loc_str='Waseca 2016',
              y_col='V10_NUP_lbac', means_sep=means_sep_W16_V10NUP)


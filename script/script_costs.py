# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:57:23 2019

@author: nigo0024
"""

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
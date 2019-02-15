# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:14:55 2019

@author: nigo0024
"""
# In[Group by rep]
import seaborn as sns

plt.style.use('ggplot')

sns.scatterplot(x='rate_n_applied_lbac', y='yld_grain_dry_buac', hue='rep', data=df_nue13wil_pre)
sns.scatterplot(x='rep', y='yld_grain_dry_buac', hue='rate_n_applied_lbac', data=df_nue13wil_pre)

df_nue13wil_pre.groupby(['rep']).mean()['yld_grain_dry_buac']

# In[Get table from dbase]
from py_epic import Dbase_tools

def replace_missing_vals(df, missing_val='.', cols_numeric=None):
    '''
    Finds missing data in pandas dataframe and replaces with np.nan
    '''
    df.replace(missing_val, np.nan, inplace=True)
#    df = df.dropna()
    if cols_numeric is not None:
        df[cols_numeric] = df[cols_numeric].apply(pd.to_numeric)
    return df

fname_db_pw = r'G:\BBE\Agrobot\Tyler Nigon\EPIC\Data\py_epicdb_key.json'
db_tools = Dbase_tools(fname_db_pw=fname_db_pw, db_default='swc',
                       schema_default='nitrogen_social_costs')

df_willmar_plots = db_tools.get_table('willmar_2013_plotbounds_data')
df_willmar_plots = replace_missing_vals(df_willmar_plots, missing_val='.', cols_numeric='j_yld_grai')


g = sns.FacetGrid(df, col='nrate')
axes = g.axes
for ax in axes:
    sns.scatterplot(x_plt, y_plt, ax=ax)
g.map(plt.scatter, 'dem_mean', 'j_yld_grai')
plt.tight_layout()

# Remove all rows where x or y is NAN
df = df_willmar_plots.dropna(subset=['dem_mean', 'j_yld_grai'])
df_0 = df.loc[df['nrate'] == 0]
df_280 = df.loc[df['nrate'] == 280]

def plot_scatter(df, x_plt='dem_mean', y_plt='j_yld_grai'):
    ax = sns.scatterplot(x=x_plt, y=y_plt, hue='rep', data=df)

    x = df[x_plt]
    y = df[y_plt]
    best_fit = np.polyfit(x, y, 1)
    str_eq = 'y = {0:.2f}x + {1:.1f}'.format(best_fit[0], best_fit[1])
    plt.plot(np.unique(x), np.poly1d(best_fit)(np.unique(x)), color='k')
    plt.text(0.05, .95, str_eq,
             transform=ax.transAxes,
             bbox=dict(boxstyle="round",
                       ec=(0.2, 0.2, 0.2),
                       fc=(0.2, 0.2, 0.2),
                       alpha=0.2
                       ))
    plt.tight_layout()

    print(str_eq)
    g = sns.FacetGrid(df, col='nrate', hue='rep', height=3, aspect=0.8)
    g = g.map(plt.scatter, x_plt, y_plt)
    axes = g.axes[0]
    for idx, ax in enumerate(axes):
        df_temp = df.loc[df['nrate'] == g.col_names[idx]]
#        sns.scatterplot(x_plt, y_plt, data=df_temp, ax=ax)

        x = df_temp[x_plt]
        y = df_temp[y_plt]
        best_fit_temp = np.polyfit(x, y, 1)
        str_eq_temp = 'y = {0:.2f}x + {1:.1f}'.format(best_fit_temp[0], best_fit_temp[1])
        ax.plot(np.unique(x), np.poly1d(best_fit_temp)(np.unique(x)), color='k')
        ax.set_title('N = {0}'.format(g.col_names[idx]))
        plt.text(0.05, .95, str_eq_temp,
                 transform=ax.transAxes,
                 bbox=dict(boxstyle="round",
                           ec=(0.2, 0.2, 0.2),
                           fc=(0.2, 0.2, 0.2),
                           alpha=0.2
                           ))
    plt.tight_layout()

plot_scatter(df, x_plt='dem_mean', y_plt='j_yld_grai')
plot_scatter(df, x_plt='twi_mean', y_plt='j_yld_grai')

# In[Split dataset into high yielding and low yielding]
missing_data = [3, 7, 11, 12]
low = [2, 4, 5, 8, 9, 13]
high = [1, 6, 10, 14, 15, 16]

df_willmar_plots['low_high'] = np.nan
df_willmar_plots.loc[df_willmar_plots['rep'].isin(low), 'low_high'] = 0
df_willmar_plots.loc[df_willmar_plots['rep'].isin(high), 'low_high'] = 1

df_willmar_low = df_willmar_plots[df_willmar_plots['low_high'] == 0]
df_willmar_high = df_willmar_plots[df_willmar_plots['low_high'] == 1]

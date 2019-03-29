#!/usr/bin/env python
# coding: utf-8

# ## Tutorial
# ### Install Python
#
# The EONR package was designed to be used with as little Python expertise as possible. However, you may find it to your benefit to become at least a little bit familiar with Python before using EONR. If you're a beginner, it will be helpful to read the [Python Tutorial](https://docs.python.org/3/tutorial/) and/or get some hands-on practice with [this Python Tutorial by w3Schools](https://www.w3schools.com/python/).
#
# *Code for all the examples is located in your* `PYTHONPATH/Lib/site-packages/eonr/examples` *folder. With that said, you should be able to make use of* `EONR` *by following and executing the commands in this tutorial, with the only exception that you substitute in your data.*
#
# *You will find the following code included in the* `quick_start.py` *or* `quick_start.ipynb` *(for [Jupyter notebooks](https://jupyter.org/)) files in your* `PYTHONPATH/Lib/site-packages/eonr/examples` *folder - feel free to load that into your Python IDE to follow along.*

# - - -
# ### Load modules
# After [installation](installation.md), load `Pandas` and the `EONR` module in a Python interpreter:

# In[1]:


import pandas as pd
from eonr import EONR


# - - -
# ### Load the data
# `EONR` uses Pandas dataframes to access and manipulate the experimental data.

# In[2]:


df_data = pd.read_csv(r'data\minnesota_2012.csv')
df_data


# - - -
# ### Set column names *(pre-init)*
# *The table containing the experimental data **must** have a minimum of two columns:*
# * *Nitrogen fertilizer rate*
# * *Grain yield*
#
# `EONR` accepts custom column names. Just be sure to set them by either using `EONR.set_column_names()` or by passing them to `EONR.calculate_eonr()`. We will declare the names of the these two columns as they exist in the `Pandas` dataframe so they can be passed to `EONR` later:

# In[3]:


col_n_app = 'rate_n_applied_kgha'
col_yld = 'yld_grain_dry_kgha'


# Each row of data in our dataframe should correspond to a nitrogen rate treatment plot. It is common to have several other columns describing each treatment plot (e.g., year, location, replication, nitrogen timing, etc.). These aren't necessary, but `EONR` will try pull information from "year", "location", and "nitrogen timing" for labeling the plots that are generated (see [Plotting](../../docs/build/html/plotting.html) for more information).
#
# - - -
# ### Set units
# Although optional, it is good practice to declare units so we don't get confused:

# In[4]:


unit_currency = '$'
unit_fert = 'kg'
unit_grain = 'kg'
unit_area = 'ha'


# These unit variables are only used for plotting (titles and axes labels), and they are not actually used for any computations.
#
# - - -
# ### Set economic conditions
# `EONR` computes the _**Economic** Optimum Nitrogen Rate_ for any economic scenario that we define. All that is required is to declare the cost of the nitrogen fertilizer (per unit, as defined above) and the price of grain (also per unit). Note that the cost of nitrogen fertilizer can be set to zero, and the _**Agronomic** Optimum Nitrogen Rate_ will be computed.

# In[5]:


cost_n_fert = 0.88  # in USD per kg nitrogen
price_grain = 0.157  # in USD per kg grain


# - - -
# ### Initialize `EONR`
# At this point, we can initialize an instance of `EONR`.
#
# *Before doing so, we may want to set the base directory.* `EONR.base_dir` *is the default location for saving plots and data processed by* `EONR`*. If *`EONR.base_dir` *is not set, it will be set to be a folder named "eonr_temp_out" in the current working directory during the intitialization (to see your current working directory, type* `os.getcwd()`*). If you do not wish to use this as your current working directory, it can be passed to the* `EONR` *instance using the* `base_dir` *keyword.*
#
# For demonstration purposes, we will set `EONR.base_dir` to what would be the default folder if nothing were passed to the `base_dir` keyword --> that is, we will choose a folder named "eonr_temp_out" in the current working directory (`EONR` will create the directory if it does not exist).
#
# And finally, to create an instance of `EONR`, pass the appropriate variables to `EONR()`:

# In[6]:


import os
base_dir = os.path.join(os.getcwd(), 'eonr_temp_out')

my_eonr = EONR(cost_n_fert=cost_n_fert,
               price_grain=price_grain,
               col_n_app=col_n_app,
               col_yld=col_yld,
               unit_currency=unit_currency,
               unit_grain=unit_grain,
               unit_fert=unit_fert,
               unit_area=unit_area,
               base_dir=base_dir)


# - - -
# ### Calculate the EONR
# With `my_eonr` initialized as an instance of `EONR`, we can now calculate the economic optimum nitrogen rate by calling the `calculate_eonr()` method and passing the dataframe with the loaded data:

# In[7]:


my_eonr.calculate_eonr(df_data)


# It may take several seconds to run - this is because it computes the profile-likelihood and bootstrap confidence intervals by default (and as described in the [Background section](#background) this is the real novelty of `EONR` package).

# And that's it! The economic optimum for this dataset and economic scenario was **162 kg nitrogen per ha** (with 90% confidence bounds at **131** and **208 kg per ha**) and resulted in a maximum net return of nearly **$770 per ha**.
#
# - - -
# ### Plotting the EONR
# This is great, but of course it'd be useful to see our data and results plotted. Do this by calling the ```plot_eonr()``` module and *(optionally)* passing the minimum/maximum values for each axis:

# In[8]:


my_eonr.plot_eonr(x_min=-5, x_max=300, y_min=-100, y_max=1400)


# * The blue points are _**experimental data**_ (yield value in \\$ per ha as a function of nitrogen rate)
# * The blue line is the best-fit quadratic-plateau model representing _**gross return to nitrogen**_
# * The red line is the _**cost of nitrogen fertilizer**_
# * The green line is the difference between the two and represents the _**net return to nitrogen**_
# * The green point is the _**Economic Optimum Nitrogen Rate (EONR)**_
# * The transparent grey box surrounding the EONR/MRTN (green point) illustrates the _**90\% confidence intervals**_
#
# The EONR is the point on the x-axis where the net return curve (green) reaches the maximum return. The return to nitrogen at the EONR is the **Maximum Return to Nitrogen (MRTN)**, indicating the profit that is earned at the economic optimum nitrogen rate.
#
# *Notice the economic scenario (i.e., grain price, nitrogen fertilizer cost, etc.) and the "Base zero" values in the upper right corner describing the assumptions of EONR calculatioon. "Base zero" refers to the initial y-intercept of the gross return model (this setting can be turned on/off by seeting* `EONR.base_zero` *to* `True/False` *[Setting "base zero" section](quick_start#setting-base-zero) for more information).*
#
# - - -
# ### Accesing complete results
# All results (e.g., EONR, MRTN, r2 and root mean squared errors from best-fit models, confidence intervals, etc.) are stored in the `EONR.df_results` dataframe:

# In[9]:


df_results = my_eonr.df_results
df_data = my_eonr.df_data

# - - -
# ### Visualizing all confidence intervals
# By default, the confidence intervals (CIs) are calculated at many alpha levels. Noting that $\text{CI} = 1-\alpha$, let's plot the _**Wald**_ CIs, _**profile-likelihood**_ CIs, and _**bootstrap**_ CIs for a range of $\alpha$ values.

# In[10]:


my_eonr.plot_tau()


# This plot shows the lower and upper confidence intervals of the *True EONR* (*True EONR* refers to the actual EONR value, which is not actually known due to uncertainty in the dataset). At 0\% confidence, the *True EONR* is the *maximum likelihood* value, but as we increase the confidence level from 67\%, 80\%, 90\%, 95\%, and 99\%, the statistical range of the *True EONR* widens.
#
# In general, the profile-likelihood CIs are considered the most accurate of the three methods because they reflect the actual, often asymmetric, uncertainty in a parameter estimate [Cook & Weisberg, 1990](https://www.tandfonline.com/doi/abs/10.1080/01621459.1990.10476233).
#
# - - -
# ### Accessing complete CI results
# All data relating to the calculation of the CIs are saved in the `EONR.df_ci` dataframe:

# In[11]:


my_eonr.df_ci


# - - -
# ### Adjusting the economic scenario
# These results were calculated for a specific economic scenario, but the cost of fertilizer and price of grain can be adjusted to run `EONR` for another economic scenario. Just adjust the economic scenario by passing any of:
#
# * `cost_n_fert`
# * `price_grain`
# * `cost_n_social`
#
# to `EONR.update_econ()`:

# In[12]:


cost_n_fert = 1.32  # adjusted from $0.88 per kg nitrogen
my_eonr.update_econ(cost_n_fert=cost_n_fert)


# - - -
# ### Environmental observations
# You'll notice above that we can pass the `cost_n_social` variable to `EONR.update_econ()`. This is becuase `EONR` will calculate the  _**Socially Optimum Nitrogen Rate (SONR)**_ if certain environmental data are available. For more information about the _**SONR**_, refer to the [Background chapter](background.html#The-social-cost-of-nitrogen).
#
# In the same way that `cost_n_fert` was adjusted in the previous code, `cost_n_social` will be set (for the first time):

# In[13]:


cost_n_social = 1.10 # in USD per kg nitrogen
my_eonr.update_econ(cost_n_social=cost_n_social)


# - - -
# ### Set column names *(post-init)*
# You may have noticed that [the loaded data](quick_start.html#Load-the-data) for this tutorial contains columns for nitrogen uptake ("nup_total_kgha") and available nitrogen ("soil_plus_fert_n_kgha"). This data can be used to calculate the _**SONR**_ as long as the column names are correctly set.
#
# The column names were set for nitrogen fertilizer rate (`col_n_app`) and grain yield (`col_yld`) during the initialization of `EONR`, but they haven't been set for the nitrogen uptake or available nitrogen columns. This can be done (even after initilization of `EONR`) using `EONR.set_column_names()`:

# In[14]:


col_crop_nup = 'nup_total_kgha'
col_n_avail = 'soil_plus_fert_n_kgha'

my_eonr.set_column_names(col_crop_nup=col_crop_nup,
                         col_n_avail=col_n_avail)


# `EONR` simply subtracts *end of season total nitrogen uptake* from *available nitrogen* to get _**net crop nitrogen use**_, which is subsequently used to calculate the _**SONR**_.
#
# - - -
# ### Run `EONR` for the socially optimum rate
# Then simply run `EONR.calculate_eonr()` again to calculate the _**SONR**_ for the updated economic scenario:

# In[15]:


my_eonr.calculate_eonr(df_data)


# The new results are appended to the `EONR.df_results` dataframe:

# In[16]:


my_eonr.df_results


# `EONR.plot_eonr()` and `EONR.plot_tau()` can be called again to plot the new results:

# In[17]:


my_eonr.plot_eonr(x_min=-5, x_max=300, y_min=-200, y_max=1400)
my_eonr.plot_tau()


# Notice the added data in the nitrogen response plot:
# * The gold points represent _**net crop nitrogen use**_ (expressed as a \\$ amount based on the value set for `cost_n_social`)
# * The gold line is the best-fit exponential model representing _**net crop nitrogen use**_ (`EONR` fits both a linear and exponential model for this, then uses whichever has a higher $\text{r}^2$)
#
# - - -
# ### Saving the data
# The results generated by `EONR` can be saved to the `EONR.base_dir` using the `Pandas` `df.to_csv()` function. A folder will be created in the base_dir whose name is determined by the _**current economic scenario**_ of `my_eonr` (in this case "social_154_1100", corresponding to `cost_n_social > 0`, `price_ratio = 15.4`, and `cost_n_social = 1.10`, respectively):

# In[18]:


print(my_eonr.base_dir)

my_eonr.df_results.to_csv(os.path.join(os.path.split(my_eonr.base_dir)[0], 'quick_start_results.csv'), index=False)
my_eonr.df_ci.to_csv(os.path.join(os.path.split(my_eonr.base_dir)[0], 'quick_start_ci.csv'), index=False)


# Upon generating figures using `EONR.plot_eonr()` or `EONR.plot_tau()`, the `matplotlib` figures are stored to the `EONR` class. They can be saved to file by using `EONR.plot_save()`:

# In[19]:


fname_eonr_plot = 'eonr_mn2012_pre.png'
fname_tau_plot = 'tau_mn2012_pre.png'

my_eonr.plot_save(fname=fname_eonr_plot, fig=my_eonr.fig_eonr)
my_eonr.plot_save(fname=fname_tau_plot, fig=my_eonr.fig_tau)


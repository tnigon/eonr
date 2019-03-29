#!/usr/bin/env python
# coding: utf-8

# ## Advanced Tutorial
# ### Follow along
# 
# Code for all the examples is located in your `PYTHONPATH/Lib/site-packages/eonr/examples` folder. With that said, you should be able to make use of `EONR` by following and executing the commands in this tutorial using either the sample data provided or substituting in your own data.
# 
# *You will find the following code included in the* `advanced_tutorial.py` *or* `advanced_tutorial.ipynb` *(for [Jupyter notebooks](https://jupyter.org/)) files in your* `PYTHONPATH/Lib/site-packages/eonr/examples` *folder - feel free to load that into your Python IDE to follow along.*
# 
# - - -
# ### Calculate `EONR` for several economic scenarios 
# In this tutorial, we will run `EONR.calculate_eonr()` in a loop, adjusting the economic scenario prior to each run.

# - - -
# ### Load modules
# Load `pandas` and `EONR`:

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
# ### Set column names and units
# *The table containing the experimental data **must** have a minimum of two columns:*
# * *Nitrogen fertilizer rate*
# * *Grain yield*
# 
# We'll also set *nitrogen uptake* and *available nitrogen* columns right away for calculating the socially optimum nitrogen rate.
# 
# As a reminder, we are declaring the names of these columns and units because they will be passed to `EONR` later.

# In[3]:


col_n_app = 'rate_n_applied_kgha'
col_yld = 'yld_grain_dry_kgha'
col_crop_nup = 'nup_total_kgha'
col_n_avail = 'soil_plus_fert_n_kgha'

unit_currency = '$'
unit_fert = 'kg'
unit_grain = 'kg'
unit_area = 'ha'


# - - -
# ### Turn `base_zero` off
# You might have noticed the `base_zero` option for the `EONR` class in the [API](my_eonr.html#module-eonr.eonr). `base_zero` is a `True`/`False` flag that determines if gross return to nitrogen should be expressed as an absolute values. We will see a bit later that upon executing `EONR.calculate_eonr()`, grain yield from the input dataset is used to create a new column for gross return to nitrogen *("grtn")* by multiplying the grain yield column by the price of grain (`price_grain` variable). 
# 
# If `base_zero` is `True` *(default)*, the observed yield return data are standardized so that the best-fit quadratic-plateau model passes through the y-axis at zero. This is done in two steps:
# 1. Fit the quadratic-plateau to the original data to determine the value of the y-intercept of the model ($\beta_0$)
# 2. Subtract $\beta_0$ from all data in the recently created *"grtn"* column (temporarily stored in `EONR.df_data`)
# 
# This behavior (`base_zero = True`) is the default in `EONR`. However, `base_zero` can simply be set to `False` during the initialization of `EONR`. We will set it store it in its own variable now, then pass to `EONR` during initialization:

# In[4]:


base_zero = False


# - - -
# ### Initialize `EONR`
# Let's set the base directory and initialize an instance of `EONR`, setting `cost_n_fert = 0` and `price_grain = 1.0` (\\$1.00 per kg) as the default values (we will adjust them later on in the tutorial):

# In[5]:


import os
base_dir = os.path.join(os.getcwd(), 'eonr_advanced_tutorial')

my_eonr = EONR(cost_n_fert=0,
               price_grain=1.0,
               col_n_app=col_n_app,
               col_yld=col_yld,
               col_crop_nup=col_crop_nup,
               col_n_avail=col_n_avail,
               unit_currency=unit_currency,
               unit_grain=unit_grain,
               unit_fert=unit_fert,
               unit_area=unit_area,
               base_dir=base_dir,
               base_zero=base_zero)


# - - -
# ### Calculate the *AONR*
# You may be wondering why `cost_n_fert` was set to 0. Well, setting our nitrogen fertilizer cost to $0 essentially allows us to calculate the optimum nitrogen rate ignoring the cost of the fertilizer input. This is known as the *Agronomic Optimum Nitrogen Rate (AONR)*. The AONR provides insight into the maximum achievable grain yield. Notice `price_grain` was set to `1.0` - this effectively calculates the AONR so that the maximum return to nitrogen (MRTN), which will be expressed as \\$ per ha when ploting via `EONR.plot_eonr()`, is similar to units of kg per ha (the units we are using for grain yield).
# 
# Let's calculate the AONR and plot it (adjusting `y_max` so it is greater than our maximum grain yield):

# In[6]:


my_eonr.calculate_eonr(df_data)
my_eonr.plot_eonr(x_min=-5, x_max=300, y_min=-100, y_max=18000)


# We see that the _**agronomic**_ optimum nitrogen rate was calculated as **177** kg per ha, and the MRTN is **13.579 Mg per ha** (yes, it says $13,579, but because `price_grain` was set to \\$1, the values are equivalent and the units can be substituted.
# 
# If you've gone through the [first tutorial](tutorial.md), you'll notice there are a few major differences in the look of this plot:
# 
# **The red line representing nitrogen fertilizer cost is on the bottom of the plot**
# *This happened because* `cost_n_fert` *was set to zero, and nitrogen fertilizer cost is simply being plotted as a flat horizontal line at $\text{y}=0$*
# 
# 
# **The blue line is missing?**
# *The gross return to nitrogen (GRTN) line representing the best-fit of the quadratic-plateau model (blue line) is there, but it is actually being covered up by the green line (net return to nitrogen; NRTN). This is the case because `cost_n_fert` was set to zero.
# 
# **The GRTN/NRTN lines do not pass through the y-intercept at $\text{y}=0$?**
# 
# *Because* `base_zero` *was set to* `False`, *the observed data (blue points) were not standardized as to *force* the best-fit quadratic-plateau model from passing through at* $\text{y}=0$.

# - - -
# ### Loop through several economic conditions
# `EONR` computes the optimum nitrogen rate for any economic scenario that we define. The `EONR` class is designed so the economic condidtions can be adjusted, calculating the optimum nitrogen rate after each adjustment. We just have to set up a simple loop to update the economic scenario (using `EONR.update_econ()`) and calculate the EONR (using `EONR.calculate_eonr()`).
# 
# We will also generate plots and save them to our base directory right away:

# In[7]:


price_grain = 0.157  # set from $1 to 15.7 cents per kg
cost_n_fert_list = [0.44, 0.88, 1.32, 1.76, 2.20]

for cost_n_fert in cost_n_fert_list:
    # first, update fertilizer cost
    my_eonr.update_econ(cost_n_fert=cost_n_fert,
                        price_grain=price_grain)
    # second, calculate EONR
    my_eonr.calculate_eonr(df_data)
    # third, generate (and save) the plots
    my_eonr.plot_eonr(x_min=-5, x_max=300, y_min=-100, y_max=2800)
    my_eonr.plot_tau()
    my_eonr.plot_save(fname='eonr_mn2012_pre.png', fig=my_eonr.fig_eonr)
    my_eonr.plot_save(fname='tau_mn2012_pre.png', fig=my_eonr.fig_tau)    


# A similar loop can be made adjusting the social cost of nitrogen. `my_eonr.base_zero` is set to `True` to compare the graphs:

# In[8]:


price_grain = 0.157  # keep at 15.7 cents per kg
cost_n_fert = 0.88  # set to be constant
my_eonr.update_econ(price_grain=price_grain,
                    cost_n_fert=cost_n_fert)
my_eonr.base_zero = True  # let's use base zero to compare graphs

cost_n_social_list = [0.44, 0.88, 1.32, 2.20, 4.40]

for cost_n_social in cost_n_social_list:
    # first, update social cost
    my_eonr.update_econ(cost_n_social=cost_n_social)
    # second, calculate EONR
    my_eonr.calculate_eonr(df_data)
    # third, generate (and save) the plots
    my_eonr.plot_eonr(x_min=-5, x_max=300, y_min=-400, y_max=1400)
#     my_eonr.plot_tau()
    my_eonr.plot_save(fname='eonr_mn2012_pre.png', fig=my_eonr.fig_eonr)
#     my_eonr.plot_save(fname='tau_mn2012_pre.png', fig=my_eonr.fig_tau)


# 
# Notice that we used the same `EONR` instance (`my_eonr`) for all runs. This can be done if there are mnay combinations of economic scenarios, or many expermintal datasets, that you'd like to loop through. If you'd like the results to be saved separate (perhaps to separate results depending if a social cost is considered) that's fine too. Simply create a new instance of `EONR` and customize how you'd like.
# - - -
# ### View results
# All eleven runs can be viewed in the dataframe:

# In[9]:


my_eonr.df_results


# Notice the *"price_ratio"*, *"eonr"*, *"mrtn"*, *etc.* are different for each scenario. 
# - - -
# ### Save the data
# The results generated by `EONR` can be saved to the `EONR.base_dir` using the `Pandas` `df.to_csv()` function. A folder will be created in the base_dir whose name is determined by the _**current economic scenario**_ of `my_eonr` (in this case "social_336_4400", corresponding to `cost_n_social > 0`, `price_ratio == 33.6`, and `cost_n_social == 4.40` for "social", "336", and "4400" in the folder name, respectively):

# In[10]:


print(my_eonr.base_dir)

my_eonr.df_results.to_csv(os.path.join(os.path.split(my_eonr.base_dir)[0], 'advanced_tutorial_results.csv'), index=False)
my_eonr.df_ci.to_csv(os.path.join(os.path.split(my_eonr.base_dir)[0], 'advanced_tutorial_ci.csv'), index=False)


# -*- coding: utf-8 -*-
# In[1: load Pandas and the EONR module]
import pandas as pd
from eonr import EONR

# In[2: Load the data]
df_data = pd.read_csv(r'data\minnesota_2012.csv')

# In[3: Declare the names of the nitrogen rate and grain yield columns]
col_n_app = 'rate_n_applied_kgha'
col_yld = 'yld_grain_dry_kgha'

# In[4: Declare the units being used]
unit_currency = '$'
unit_fert = 'kg'
unit_grain = 'kg'
unit_area = 'ha'

# In[5: Declare cost of nitrogen and price of grain]
cost_n_fert = 0.88  # in USD per kg nitrogen
price_grain = 0.157  # in USD per kg grain


# In[6: Initialize an instance of EONR]
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

# In[7: Calculate the EONR]
my_eonr.calculate_eonr(df_data)

# In[8: Plot the results]
my_eonr.plot_eonr(x_min=-5, x_max=300, y_min=-100, y_max=1400)

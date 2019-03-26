## Quick Start

The EONR package was designed to be used with as little Python expertise as possible. However, you may find it to your benefit to become at least a little bit familiar with Python before using EONR. If you're a beginner, I recommend reading the [basic Python Tutorial](https://docs.python.org/2/tutorial/ " Basic Python tutorial") or [Think Python](http://www.greenteapress.com/thinkpython/ "Think Python").

With that said, you should be able to make use of EONR by following and executing the commands in the basic tutorial, with the only exception that you substitute in your data.

Code for all the examples is located in your `PYTHONPATH/Lib/site-packages/eonr/examples` folder.

You will find the following code included in the `quick_start.py` or `quick_start.ipynb` (for [Jupyter notebooks](https://jupyter.org/)) files in your `PYTHONPATH/Lib/site-packages/eonr/examples` folder - feel free to load that into your Python IDE to follow along.

After [installation](installation.md), load `Pandas` and the `EONR` module in a Python interpreter:
```python
import pandas as pd
from eonr import EONR
```

Load the data. `EONR` uses Pandas dataframes to access and manipulate the experimental data.

```python
df_data = pd.read_csv(r'examples\minnesota_2012.csv')
df_data.head()
```

**Figure 1:** Dataframe containing nitrogen rate, grain yield, and nitrogen uptake data from a field trial in Minnesota.
![](img/quick_start_minnesota_dataframe.png "Minnesota 2012 EONR Data")

The table containing the experimental data **must** have a minimum of two columns:
* Nitrogen fertilizer rate
* Grain yield

`EONR` accepts custom column names. Just be sure to set them by either using `EONR.set_column_names()` or by passing them to `EONR.calculate_eonr()`. We will declare the names of the these two columns as they exist in the `Pandas` dataframe so they can be passed to `EONR` later:
```python
col_n_app = 'rate_n_applied_kgha'
col_yld = 'yld_grain_dry_kgha'
```

Each row of data in our dataframe should correspond to a nitrogen rate treatment plot. It is common to have several other columns describing each treatment plot (e.g., year, location, replication, nitrogen timing, etc.). These aren't necessary, but `EONR` will try pull information from "year", "location", and "nitrogen timing" for labeling the plots that are generated (see [Plotting](plotting.md) for more information).

Although optional, it is good practice to declare units so we don't get confused:

```python
unit_currency = '$'
unit_fert = 'kg'
unit_grain = 'kg'
unit_area = 'ha'
```

These unit variables are only used for plotting (titles and axes labels), and they are not actually used for any computations.

`EONR` computes the _**Economic** Optimum Nitrogen Rate_ for any economic scenario that we define. All that is required is to declare the cost of the nitrogen fertilizer (per unit, as defined above) and the price of grain (also per unit). Note that the cost of nitrogen fertilizer can be set to zero, and the _**Agronomic** Optimum Nitrogen Rate_ will be computed.

```python
cost_n_fert = 0.88  # in USD per kg nitrogen
price_grain = 0.157  # in USD per kg grain
```

At this point, we can initialize an instance of `EONR`.

Before doing so, we may want to set the base directory. `EONR.base_dir` is the default location for saving plots and data processed by `EONR`. If `EONR.base_dir` is not set, it will be set to be a folder named "eonr_temp_out" in the current working directory during the intitialization (to see your current working directory, type `os.getcwd()`). If you do not wish to use this as your current working directory, it can be passed to the `EONR` instance using the `base_dir` keyword.

For demonstration purposes, we will set `EONR.base_dir` to what would be the default folder if nothing were passed to the `base_dir` keyword --> that is, we will choose a folder named "eonr_temp_out" in the current working directory (`EONR` will create the directory if it does not exist).

And finally, to create an instance of `EONR`, pass the appropriate variables to `EONR()`:

```python
import os
base_dir = os.path.join(os.getcwd(), 'eonr_temp_out')

my_eonr = EONR(cost_n_fert=cost_n_fert,
               price_grain=price_grain,
               col_n_app=col_n_app,
               col_yld=col_yld,
               unit_currency=unit_currency,
               unit_grain=unit_grain,
               unit_fert=unit_fert,
							 base_dir=base_dir,
               unit_area=unit_area)
```

With `my_eonr` initialized as an instance of `EONR`, we can now calculate the economic optimum nitrogen rate by calling the `calculate_eonr()` method and passing the dataframe with the loaded data:

```python
my_eonr.calculate_eonr(df_data)
```

It may take several seconds to run - this is because it computes the profile-likelihood and bootstrap confidence intervals by default (and as described in the [Background page](background.md) this is the real novelty of `EONR` package).

	Computing EONR for Minnesota 2012 Pre
	Cost of N fertilizer: $0.88 per kg
	Price grain: $0.16 per kg
	Economic optimum N rate (EONR): 162.3 kg per ha [130.5, 207.8] (90.0% confidence)
	Maximum return to N (MRTN): $767.93 per ha

And that's it! The economic optimum for this dataset and economic scenario was **162 kg nitrogen per ha** (with 90% confidence bounds at **131** and **208 kg per ha**) and resulted in a maximum net return of nearly **$770 per ha**. This is great, but of course it'd be useful to see our data and results plotted. Do this by calling the ```plot_eonr()``` module and passing the minimum/maximum values for each axis (optional):

```python
my_eonr.plot_eonr(x_min=-5, x_max=300, y_min=-100, y_max=1400)
```
**Figure 2:** The plotted results - blue points are experimental data (yield value in USD as a function of nitrogen rate), the blue line is the best-fit quadratic-plateau model representing gross return to nitrogen, the red line is the cost of nitrogen fertilizer, and the green line is the difference between the two and represents the net return to nitrogen.

![](img/quick_start_eonr_2012_mn.png "quick-start-eonr-plot")

 The point on the x-axis where the net return curve (green) reaches the maximum return is the **Economic Optimum Nitrogen Rate (EONR)**. The return to nitrogen at that maximum point is the **Maximum Return to Nitrogen (MRTN)**, indicating the profit that is earned at the economic optimum nitrogen rate. The 90% confidence intervals are illustrated as a transparent grey box surrounding the EONR/MRTN point.

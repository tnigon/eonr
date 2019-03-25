# EONR - Economic Optimum Nitrogen Rate Tool

*A Python tool for computing the optimum nitrogen rate and its confidence intervals from agricultural research data*

*Note: To use a table of contents tree, I have to use the index.rst file..*

## Introduction

EONR is a Python package for computing the economic optimum nitrogen fertilizer rate using data from agronomic field trials. The concept behind the economic optimum nitrogen rate approach (also referred to as the *Maximum Return to Nitrogen* approach) is to make the most favorable nitrogen fertilizer recommendation considering three variables:
* Grain price ($ per kg)
* Fertilizer cost ($ per kg)
* Grain yield response to nitrogen fertilizer (modeled from input data)
For more information about how the economic optimum nitrogen rate is calculated, see the [Background](#background) section.

## Quick Start
[*back to top*](#introduction)

To start using the EONR package, you need to have EONR installed. To install the latest release of EONR, you can use pip:

	pip install eonr

Please see the [Installation section](#installation) for more options.

You will find the following code included in the `quick_start.py` file in the `/examples` folder - feel free to load that into your Python IDE to follow along.

In a Python interpreter, load Pandas and the EONR module:
```python
import pandas as pd
from eonr import EONR
```

Load the data. EONR uses Pandas dataframes to access and manipulate the experimental data.

```python
df_data = pd.read_csv(r'examples\minnesota_2018.csv')
```

The table containing the experimental data **must** have a minimum of two columns:
* Nitrogen fertilizer rate
* Grain yield

We must declare the names of the these two columns as they exist in the Pandas dataframe (will be passed to EONR later):
```python
col_n_app = 'rate_n_applied_kgha'
col_yld = 'yld_grain_dry_kgha'
```

Each row of data in our dataframe should correspond to a nitrogen rate treatment plot. It is common to have several other columns describing each treatment plot (e.g., year, location, replication, timing, etc.). These aren't necessary, but EONR will pull information from year and location for labeling the plots that are generated (see [Plotting](plotting.md) for more information).

Although optional, it is good practice to declare units:

```python
unit_currency = '$'
unit_fert = 'kg'
unit_grain = 'kg'
unit_area = 'ha'
```

These unit variables are only used for plotting (titles and axes labels), and they are not actually used for any computations.

EONR computes the _**Economic** Optimum Nitrogen Rate_ for any economic scenario that we define. All that is required is to declare the cost of the nitrogen fertilizer (per unit, as defined above) and the price of grain (also per unit). Note that the cost of nitrogen fertilizer can be set to zero, and the _**Agronomic** Optimum Nitrogen Rate_ will be computed.

```python
cost_n_fert = 0.88  # in USD per kg nitrogen
price_grain = 0.157  # in USD per kg grain
```

At this point, we can initialize an instance of EONR, passing the appropriate variables:

```python
my_eonr = EONR(cost_n_fert=cost_n_fert,
               price_grain=price_grain,
               col_n_app=col_n_app,
               col_yld=col_yld,
               unit_currency=unit_currency,
               unit_grain=unit_grain,
               unit_fert=unit_fert,
               unit_area=unit_area)
```

With `my_eonr` initialized as an instance of EONR, we can now calculate the economic optimum nitrogen rate by calling the `calculate_eonr()` method and passing the dataframe with the loaded data:

```python
my_eonr.calculate_eonr(df_data)
```

It may take several seconds to run - this is because it computes the profile-likelihood and bootstrap confidence intervals by default (and as described in the [Background section](#background) this is the real novelty of EONR package).

	Computing EONR for Minnesota 2018 Pre
	Cost of N fertilizer: $0.88 per kg
	Price grain: $0.16 per kg
	Economic optimum N rate (EONR): 162.4 kg per ha [130.5, 207.9] (90.0% confidence)
	Maximum return to N (MRTN): $770.37 per ha

And that's it! The economic optimum for this dataset and economic scenario was **162 kg nitrogen per ha** (with 90% confidence bounds at **131** and **208 kg per ha**) and resulted in a maximum net return of over **$770 per ha**. This is great, but of course it'd be useful to see our data and results plotted. Do this py calling the ```plot_eonr()``` module and passing the minimum/maximum values for each axis (optional):

```python
my_eonr.plot_eonr(x_min=-5, x_max=300, y_min=-100, y_max=1400)
```

![The plotted results - blue points are experimental data (yield value in USD as a function of nitrogen rate), the blue line is the best-fit quadratic plateau model representing gross return to nitrogen, the red line is the cost of nitrogen fertilizer, and the green line is the difference between the two and represents the net return to nitrogen. The point on the x-axis where the net return curve (green) reaches the maximum return is the *Economic Optimum Nitrogen Rate (EONR)*. The return to nitrogen at that maximum point is the *Maximum Return to Nitrogen (MRTN)*, indicating the profit that is earned at the economic optimum nitrogen rate. The 90% confidence intervals are illustrated as a transparent grey box surrounding the EONR/MRTN point.](img/quick_start_eonr_2018_mn.png "Quick Start EONR Plot")

## Background
[*back to top*](#introduction)

## Features
[*back to top*](#introduction)


## Installation
[*back to top*](#introduction)

EONR is an open-source package written in pure Python. It runs on all major platforms (i.e., Windows, Linux, Mac).

To install the latest release of EONR, you can use pip:

	pip install eonr

It’s also possible to install the released version using conda:

	conda install eonr

Alternatively, you can use pip to install the development version directly from github:

	pip install git+https://github.com/tnigon/eonr

Another option is to clone the github repository and install from your local copy. After navigating to the directory of your local copy:

	pip install .

The recommended folder directory for the EONR package is in the site-packages folder in your Python Path (alongside all other Python packages).

### Dependencies
[*back to top*](#introduction)

EONR requires the following packages:

* [\*Matplotlib](http://matplotlib.org/ "Matplotlib")
* [\*NumPy](http://www.numpy.org/ "Numpy")
* [\*Pandas](http://pandas.pydata.org/ "Pandas")
* [Scikits-Bootstrap](https://pypi.org/project/scikits.bootstrap/ "Scikits-Bootstrap")
* [\*Scipy](http://www.scipy.org/ "Scipy")
* [\*Seaborn](https://seaborn.pydata.org/ "Seaborn")
* [Uncertainties](https://pythonhosted.org/uncertainties/ "Uncertainties")

*\*These packages come pre-installed from the following Python distributions:*
* [Python xy](https://code.google.com/p/pythonxy/ "Python xy") version >2.7
* [WinPython](http://winpython.sourceforge.net/ "WinPython") version >2.7

## Troubleshooting
[*back to top*](#introduction)
Please report any issues you encounter through the [Github issue tracker](https://github.com/tnigon/eonr).


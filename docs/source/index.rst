
===================
EONR Documentation
===================

*A Python tool for computing the optimum nitrogen rate and its confidence intervals from agricultural research data*

.. |version_current| replace:: 1.0.0
Current version: |version_current|

##################
Table of Contents
##################

*Navigate the* **EONR Documentation** *using the* **"Site"** *dropdown in the Navigation Bar*

.. toctree::
   :maxdepth: 1
   :numbered:
   :titlesonly:

   Home <index.rst>
   installation.ipynb
   tutorial.ipynb
   advanced_tutorial.ipynb
   background.ipynb
   my_eonr.rst
   license.ipynb


##################
Troubleshooting
##################
Please report any issues you encounter through the `Github issue tracker <https://github.com/tnigon/eonr/issues>`_.

#######
About
#######
``EONR`` is a Python package for computing the economic optimum nitrogen fertilizer rate using data from agronomic field trials under economic conditions defined by the user (i.e., grain price and fertilizer cost).
It can be used for any crop (e.g., corn, wheat, potatoes, etc.), but the current version (|version_current|) only supports use of the quadratic-plateau piecewise model.

**Therefore, use caution in making sure that a quadratic-plateau model is appropriate for your application.**

*Future versions should add support for other models (quadratic, spherical, etc.) that may improve the fit of experimental yield response to nitrogen for other crops.*

Data requirements
******************
The minimum data requirement to utilize this package is observed (or simulated) experimental data of agronomic yield response to nitrogen fertilizer.
In other words, your experiment should have multiple nitrogen rate treatments, and you should have measured the yield for each experimental plot at the end of the season.
Suitable experimental design for your particular experiment is always suggested (e.g., it should probably be replicated).

Intended audience
******************
The intended audiences for this package are agricultural researchers, private sector organizations and consultants that support farmers, and of course those inquisitive farmers that always want to know more about their soils and the environment around them.

Concept of the EONR
********************
The concept behind the *Economic Optimum Nitrogen Rate* approach (also referred to as the *Maximum Return to Nitrogen* approach) is to make the most favorable nitrogen fertilizer recommendation considering three variables:

* Grain price ($ per kg)
* Fertilizer cost ($ per kg)
* Grain yield response to nitrogen fertilizer (modeled from input data)


.. image:: _images/intro_diagram_grey.png
   :width: 1120
   :align: left
   :alt: Corn nitrogen rate response experiment in Minnesota (photo captured in July when the crop is about shoulder-high).


On the left is a corn nitrogen rate response experiment in Minnesota (photo captured in July when the crop is about shoulder-high).
Notice the different shades of green in the crop canopy - the dark, lush green is indicative of sufficient nitrogen availability and the lighter green is indicative of nitrogen stress.
The ``EONR`` Python package was used to compute the economic optimum nitrogen rate (and its 90% confidence intervals) using experimental data, as illustrated in the plot on the right.

For more information about how the economic optimum nitrogen rate is calculated, see the `Background section <background.html>`__.

Motivation for development of ``EONR``
*************************************
Although calculation of the economic optimum nitrogen rate (EONR) from a nitrogen response experiment is a trivial task for agronomic researchers, the computation of its confidence intervals are not.
This is especially true for calculating confidence intervals for data that are explained best with a quadratic-plateau model, which is generally thought of as the most appropriate model for describing yield response to nitrogen in corn.
With the ``EONR`` package available and accessible, I hope all published EONR research also reports the confidence intervals of the maximum likelihood EONR.
Furthermore, I hope this package enables researchers and farmers to take a closer look at evaluating what *the best* nitrogen rate may be.
The EONR considers the cost of nitrogen in addition to the price a farmer receives for their grain. This is great, but this package takes this concept one step further with an added *social cost of nitrogen*.
To consider the environmental or social effect of nitrogen application in agriculture, two things are necessary:

* We have to make it a habit to measure total crop nitrogen uptake (or at least residual soil nitrogen) at the end of the season.
* As a society we have to do a better job of putting a value on the cost of pollution caused by nitrogen fertilizers.

This second point is especially tricky because it is *very subjective* and everyone will have a different opinion.
It's a complex question whose answer changes not only from watershed to watershed, but from household to household, and perhaps even within a household.

Although it is important to recognize that nitrogen probably has some social cost, it is just as important to figure out who pays for that cost. Just remember, farmers farm to produce food and earn a living, they don't farm to pollute the water and air.
Sure, they definitely bear a lot of responsibility in managing their land and their inputs, but that doesn't mean they should also bear all the costs.
If we as a society recognize that pollution caused by nitrogen fertilizer in agriculture is indeed a problem, we should work together to figure out how to support the farmers to help fix the problem (or at least stop it from getting worse).

After all, **farmers farm to grow food, they don't farm to pollute**, *right?*


###################
Indices and tables
###################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

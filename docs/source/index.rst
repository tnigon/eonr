##################
EONR Documentation
##################

*A Python tool for computing the optimum nitrogen rate and its confidence intervals from agricultural research data*

*****************
Table of Contents
*****************

*Navigate the EONR Documentation using the* **"Site"** *dropdown in the Navigation Bar*

.. toctree::
   :maxdepth: 1
   :numbered:
   :titlesonly:

   installation.md
   quick_start.md
   quick_start_jupyter.md
   quick_start_jupyter.rst
   background.md
   tutorial.md
   plotting.md
   Troubleshooting.md
   license.md
   code.rst

*****
About
*****
`EONR` is a Python package for computing the economic optimum nitrogen fertilizer rate using data from agronomic field trials under economic conditions defined by the user (i.e., grain price and fertilizer cost).
The concept behind the *Economic Optimum Nitrogen Rate* approach (also referred to as the *Maximum Return to Nitrogen* approach) is to make the most favorable nitrogen fertilizer recommendation considering three variables:

* Grain price ($ per kg)
* Fertilizer cost ($ per kg)
* Grain yield response to nitrogen fertilizer (modeled from input data)

.. image:: img/intro_diagram_grey.png
  :width: 1120
  :align: left
  :alt: Corn nitrogen rate response experiment in Minnesota (photo captured in July when the crop is about shoulder-high).


On the left is a corn nitrogen rate response experiment in Minnesota (photo captured in July when the crop is about shoulder-high).
Notice the different shades of green in the crop canopy - the dark, lush green is indicative of sufficient nitrogen availability and the lighter green is indicative of nitrogen stress.
The ``EONR`` Python package was used to compute the economic optimum nitrogen rate (and its 90% confidence intervals) using experimental data, as illustrated in the plot on the right.

For more information about how the economic optimum nitrogen rate is calculated, see the `Background page <background.html>`__.

***************
Troubleshooting
***************
Please report any issues you encounter through the `Github issue tracker <https://github.com/tnigon/eonr>`_.

****************
Equations in RST
****************
.. math::

   \frac{ \sum_{t=0}^{N}f(t,k) }{N}

Indices and tables
##################
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

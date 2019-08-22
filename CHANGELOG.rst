Release: 0.2.0
***************
**Date**: 2019 August 22

**Description**: Support for a quadratic model

* If model is not specified, both quadratic and quadratic-plateau models are fit, and the model with the highest r^2 is used
* The model to be used can be specified using ``EONR.model`` (e.g., 'quadratic' or 'quad_plateau')
* Removed "bootstrap" text from the "Tau" plot if the bootstrap confidence interval is not computed and there is no bootstrap data to show.
* Added the model type to ``EONR.df_results`` and the legend of the EONR plot.

Release: 0.1.4
***************
**Date**: 2019 June 04

**Description**: Several feature additions and enhancements

* Consider fixed costs by setting ``EONR.costs_fixed``
* Added feature to use custom plot title (e.g., ``EONR.plot_modify_title("My New Title")``)
* Added feature to plot only the NRTN and cost_n_fert line if cost_n_fert is not zero
* Assert that user has EONR.price_grain > 0 before calculating EONR
* Added feature to compute the difference from the t-statistic (as a function of theta2/N rate) when calculating profile-likelihood CIs for EONR
* Added feature to plot the difference from the t-statistic as a function of theta2/N rate (e.g., ``EONR.plot_delta_tstat``)
* Included "R*" and "costs_at_onr" to ``EONR.df_results``.
* Taking precaution to be sure optimization of profile-likelihood CIs aren't caught at a local miniumum.
* Moving all "linspace" arrays to a new dataframe (``EONR.df_linspace``)
* Calculating the derivative of the net return to nitrogen curve (``EONR.coefs_nrtn``)
* Added basic plotting for a zoomed in look at the net return curve (``EONR.plot_derivative``)
* Added vertical line to represent ONR to ``EONR.fig_tau``

Release: 0.1.3
***************
**Date**: 2019 March 31

**Description**: Fix to profile-likelihood and bootstrap CIs when ``cost_n_social > 0``.

* Added examples folder to official package distribution
* Added a fix so ``EONR.models.R`` is updated for ``cost_n_social > 0``
* Column name in df_ci changed from "eonr_error" to "eonr_bias".

Release: 0.1.2
***************
**Date**: 2019 March 30

**Description**: initial release
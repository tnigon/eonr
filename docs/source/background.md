## Background

### Nitrogen fertilizer recommendations
Many crops, (e.g., corn, wheat, potatoes) depend on nitrogen fertilizer to achieve profitable yields, while nitrogen itself contributes to environmental pollution and is difficult to manage because of its elusive behavior in the soil. The concept behind the *Economic Optimum Nitrogen Rate* approach (also referred to as the *Maximum Return to Nitrogen* approach) is to make the most favorable nitrogen fertilizer recommendation considering three variables:

* Grain price ($ per kg)
* Fertilizer cost ($ per kg)
* Grain yield response to nitrogen fertilizer (modeled from input data)

As its name suggests, the *Maximum Return to Nitrogen* (MRTN) is the maximum achievable profit expected after subtracting the cost of nitrogen fertilizer. The economic optimum nitrogen rate (EONR) is the nitrogen rate where the MRTN is reached. In statistical terms, the EONR is the point where the monetary return from yield is equal to the cost of the increase in nitrogen fertilizer input. The approach uses a best-fit statistical model that utilizes input data from replicated field trials with several nitrogen rate treatments (**Figure 1**). Differences among replications from these trials can contribute to uncertainty in estimating the EONR, but the estimated EONR can be used as the baseline for which nitrogen fertilizer recommendations are made for a given location or soil.

**Figure 1:** Aerial photo of a nitrogen rate response experiment for corn (photo captured in July when the crop is about shoulder-high). The plot boundaries were rendered over the photo to visualize the individual plots more easily. This particular experiment had nine (9) nitrogen rates applied at three (3) different times during the season, and was replicated four (4) times. See below for treatment details.

![](img/quick_start_mn_corn.png "umn-corn-trial")

### The quadratic-plateau model
The quadratic-plateau model is often identified as the most appropriate model for nitrogen response in corn. It is a piecewise function that can be described as:

![](img/background_eq1_quad_plateau.png)

where ![](img/eq/eq_y.png) can represent grain yield (kg ha-1) or monetary return ($ ha-1) from grain yield, ![](img/eq/eq_x.png) represents quantity of nitrogen fertilizer applied, and ![](img/eq/eq_b0.png), ![](img/eq/eq_b1.png), and ![](img/eq/eq_b2.png) are the coefficients estimated from the experimental data assuming identically and independently distributed errors (![](img/eq/eq_e.png)).

**Figure 2:** Plot generated from the `EONR` package. The plotted results - blue points are experimental data (yield value in USD as a function of nitrogen rate), the blue line is the best-fit quadratic-plateau model representing gross return to nitrogen, the red line is the cost of nitrogen fertilizer, and the green line is the difference between the two and represents the net return to nitrogen.

![](img/quick_start_eonr_2012_mn.png "background-eonr-plot")

 The point on the x-axis where the net return curve (green) reaches the maximum return is the **EONR/MRTN**. The profile-likelihood 90% CIs are illustrated as a transparent grey box surrounding the EONR/MRTN point.

### Confidence intervals
One of the major novelties of the `EONR` package is that it calculates profile-likelihood CIs (as well as Wald and bootstrap CIs) from data fit with a quadratic-plateau model.

Quadratic models tend to be most popular in the literature, at least for describing how to calculate confidence intervals (CIs) for the EONR ([Hernandez & Mulla, 2008](https://dl.sciencesocieties.org/publications/aj/abstracts/100/5/1221); [Sela et al., 2017](https://dl.sciencesocieties.org/publications/jeq/abstracts/46/2/311)). Even in cases where a quadratic model happens to fit the observed data best, [Cerrato & Blackmer (1990)](https://www.agronomy.org/publications/aj/abstracts/82/1/AJ0820010138) imply that the idea of using the quadratic model is absurd because it predicts rapid decreases in yields when fertilizer is applied at higher than optimal rates, a trend that is not generally supported by evidence for maize. Furthermore, the quadratic model produces a systematic bias of overestimated maximum grain yield and optimum nitrogen fertilizer rate [(Bullock & Bullock, 1994)](https://www.agronomy.org/publications/aj/abstracts/86/1/AJ0860010191).

From a scientific perspective, it is widely recognized that large uncertainties exist around the estimated EONR computed from yield data and that it is essential to report CIs. Still, few examples exist in the agronomic literature where CIs are actually estimated ([Bachmaier & Gandorfer, 2009](https://www.researchgate.net/publication/225680100_A_conceptual_framework_for_judging_the_precision_agriculture_hypothesis_with_regard_to_site-specific_nitrogen_application); [Hernandez & Mulla, 2008](https://dl.sciencesocieties.org/publications/aj/abstracts/100/5/1221); [Jaynes, 2011](https://link.springer.com/article/10.1007/s11119-010-9168-3); [Sela et al., 2017](https://dl.sciencesocieties.org/publications/jeq/abstracts/46/2/311)). Of these examples, only Jaynes (2011) calculated CIs for the quadratic-plateau response function (which has generally been recognized as the most appropriate model for describing yield response to nitrogen in corn). [Hernandez & Mulla, 2008](https://dl.sciencesocieties.org/publications/aj/abstracts/100/5/1221) describe three general methods that can be used for estimating CIs about the EONR:
* the Wald CI
* the profile-likelihood based CI
* a bootstrap-derived CI

### The social cost of nitrogen
The `EONR` package also allows the user to define a __social cost of nitrogen__, which is then used in the optimum nitrogen rate calculations based on residual soil nitrogen (nitrogen fertilizer not taken up by the crop at the end of the season).

The traditional approach for calculating the EONR considers only the cost of the nitrogen fertilizer product, and does not consider other unintended costs of nitrogen application. The social cost of nitrogen, defined as the present value of monetary damages caused by an incremental increase in nitrogen, has been suggested as a method to place a value on pollution and other damages (e.g., health, quality of living, etc.) caused by nitrogen fertilizer application ([Keeler et al., 2016](http://stacks.iop.org/1748-9326/9/i=7/a=074002?key=crossref.75a91c07d59a4043a07280d01299d0d8)). Because of the complexity of the nitrogen cycle and the spatial and temporal variability associated with it, the social cost of nitrogen is extremely difficult to quantify and is fraught with uncertainty. Additionally, the basis for what might be considered a healthy environment or an acceptable quality of living is highly subjective and may change abruptly depending on many factors. The social cost of nitrogen is a straightforward concept, however, and it can be a useful method to assess the value of economic gain versus damages from agricultural production.

When total nitrogen uptake is measured in field experiments, there is an opportunity to calculate the quantity of nitrogen that we know was not utilized by the crop (residual nitrogen). `EONR` uses crop nitrogen uptake (and optionally, nitrogen in the soil at the beginning of the season) to model end-of-season residual nitrogen as a function of crop-available nitrogen at the beginning of the season (including nitrogen applied as fertilizer). This residual nitrogen value can be multiplied by a social cost (per unit residual nitrogen) to determine the monetary damages that a given experimental treatment might be contributing to pollution or other damages caused by nitrogen fertilizer application. `EONR` then adjusts the optimum nitrogen rate after also considering these social costs derived from experimental data. This is meaningful because it provides a basis for analyzing the costs of pollution and other damages caused by nitrogen fertilizer at the field scale. Although analysis at the regional scale is worthwhile, results oftentimes to not translate to the field scale.

Depending on the economic scenario defined by the user `EONR` will calculate one of the following:
1. _**Agronomic Optimum Nitrogen Rate**_ *(AONR)*: both the cost of nitrogen fertilizer and social cost of nitrogen are ignored
2. _**Economic Optimum Nitrogen Rate**_ *(EONR)*: cost of nitrogen fertilizer is considered but social cost is ignored
3. _**Socially Optimum Nitrogen Rate**_ *(SONR)*: both cost of nitrogen fertilizer and social cost of nitrogen are considered

# Resolving Sharper Fronts of the Agulhas Current Retroflection Using SWOT Altimetry

Authors: S. Coadou-Chaventon, S. Swart, G. Novelli, S. Speich


This is a repository for the data analyses and figures of Coadou-Chaventon et al. (2025) - [*Resolving Sharper Fronts of the Agulhas Current Retroflection Using SWOT Altimetry*](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025GL115203)

### Abstract
<p align="justify">
As a cutting-edge altimetry product, the Surface Water and Ocean Topography (SWOT) satellite observations require validation against in situ measurements and existing satellite data sets. During the fast-repeat phase of the mission, daily SWOT altimetry retrievals capture submesoscale ocean eddies (30 km) amidst larger mesoscale (100–200 km) structures in the dynamic Agulhas Current Retroflection region. Our results reveal that SWOT significantly enhances the resolution of the Agulhas Current's frontal features, producing sea surface height gradients that are 28% sharper compared to conventional altimetry products. Comparisons with in situ velocity observations from an underwater glider and surface drifters, further demonstrate SWOT's unparalleled capability to detect the strongest velocities and velocity gradients. These findings mark a pivotal step forward in resolving fine-scale ocean circulation from satellite altimetry, with promising implications for unveiling horizontal and vertical dynamics in western boundary currents and beyond.
</p>

### Workflow
<p align="justify">
The notebooks are organized as follows:

1. Comparison of SWOT with DUACS and SST for two given days. Highlights SWOT capacity to detect submesoscale features and more intense velocities along the Agulhas Retroflection front (Figure 1) - [01_FineScaleViews.ipynb](https://github.com/SolangeCoadou/CoadouChaventon-2025-GRL/blob/main/01_FineScaleViews.ipynb)
2. Comparison of SWOT and DUACS velocity fields. Reveals that SWOT captures stronger velocities than DUACS (Figure 2) - [02_StrongerVelocities.ipynb](https://github.com/SolangeCoadou/CoadouChaventon-2025-GRL/blob/main/02_StrongerVelocities.ipynb)
3. Comparison of SWOT and DUACS fronts gradients. Shows that SWOT resolves sharper fronts, in particular small-scale and high Ro gradients (Figures 3 and S2) - [03_SharperFronts.ipynb](https://github.com/SolangeCoadou/CoadouChaventon-2025-GRL/blob/main/03_SharperFronts.ipynb)
4. Comparison of SWOT and DUACS with in situ observations from drifters and a Seaglider. In situ observations higher than 0.5 m s are more closely correlated to SWOT compared to Duacs (Figures 4, S1 and S3) - [04_InSituComparison.ipynb](https://github.com/SolangeCoadou/CoadouChaventon-2025-GRL/blob/main/04_InSituComparison.ipynb)

All the functions are available in a separate file (functions.py) and loaded at the beginning of each of the notebooks.

We made use of T. Tranchant's Python toolbox [SwotDiag](https://github.com/treden/SwotDiag) to compute the Rossby number map on SWOT data.
To compute Monte Carlo Singular Spectrum Analysis on the drifters velocities, we relied on the [mcssa](https://github.com/VSainteuf/mcssa) python module.
</p>

### Data sources

* [SWOT altimetry](https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/global/swot-l3-ocean-products.html)

* [DUACS altimetry](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description)

* [Ekman currents](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_MYNRT_015_003/description)

* [Stokes drift](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)

* [The drifter data](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0301712)

* [The glider data](https://zenodo.org/records/15189208)

# Filter Profiles

This directory contains the filter profiles that are used to tabulate the black body response functions for each band. While the tabulator needs to be run only once per filter, and we provide tabulation for these filters, this is to keep the records for the references. Users can run the tabulator themselves for higher accuracy or higher computational performance, should they find it necessary.

## How to add a new (set of) filter profile(s)

For user-added filters, it does not need to be in this directory. It is recommended to **Finish writing this.**

## List of Built-in Filters

### Nancy Grace Roman Wide Field Instrument

Obtained from [the official github repo v1.1](https://github.com/spacetelescope/roman-technical-information/tree/v1.1). The transmittance is [defined as the quotient of the effective area by the collecting area of the mirror](https://roman-docs.stsci.edu/roman-instruments-home/wfi-imaging-mode-user-guide/wfi-design/wfi-optical-elements), which is [4.38562 m²](https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000481-BackgroundSpectra.pdf). Since the transmittance varies between different detectors, the values provided here are the average over all detectors.

Note that we only provide the filters used in High Latitude Time Domain Survey (HLTDS), because other filters are not relevant as of now. If necessary, user should provide their own filter profile and tabulate the response.

### SDSS *ugriz*

### *UBVRI* (Johnson -- Cousins -- Bessell)


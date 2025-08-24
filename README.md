# grib_utils
Utility functions for downloading and converting meteorological data in grib2 format

## ECMWF Open Data documentation
For full details on ECMWF Open Data, available datasets, formats, and usage guidelines, please consult the official documentation:  
https://www.ecmwf.int/en/forecasts/datasets/open-data

## ECMWF precipitation forecast can be previewed online
https://charts.ecmwf.int/products/medium-rain-acc?

## Filename convention
The filename structure for ECMWF OpenData total accumulated precipitation forecasts is:

```
ECMWF_total_accumulated_precipitation_forecast_base{YYYYMMDD}T{HH}Z_h{HHH}_step{STEP}.grb
```

Where:
- `base{YYYYMMDD}T{HH}Z` — base time (run/initialization date and time of the forecast, in UTC, e.g. `base20240824T00Z`)
- `h{HHH}` — forecast horizon, i.e. maximum forecast lead time in hours from the base time (e.g. `h240`)
- `step{STEP}` — forecast time step, i.e. interval between successive forecast outputs, in hours (e.g. `step24`)

**Example:**
```
ECMWF_total_accumulated_precipitation_forecast_base20240824T00Z_h240_step24.grb
```

## Metadata CSV contents

For each raster file, the following metadata is stored:

- `filename` — name of the raster file
- `variable` — meteorological variable (e.g. total_precipitation)
- `basetime` — combined base date and time (e.g. `20240824T00`)
- `horizon` — forecast horizon in hours
- `steps` — list of all forecast time steps (e.g. `[24, 48, 72, ...]`)
- `crs` — coordinate reference system (e.g. `EPSG:4326`)
- `lats` — list of latitude values for the raster grid
- `lons` — list of longitude values for the raster grid
- `width` — number of columns in the raster
- `height` — number of rows in the raster
- `xmin`, `ymin`, `xmax`, `ymax` — spatial extent of the raster
- `download_time` — timestamp when the metadata entry was added


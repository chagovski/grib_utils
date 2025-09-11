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

## Create the Conda Environment from environment.yml

1. Open Anaconda Prompt or your terminal.
2. Navigate to the root folder containing `environment.yml`. For example:
   ```sh
   cd path/to/grib_utils
   ```
3. Run the following command to create the environment:
   ```sh
   conda env create -f environment.yml
   ```
4. Activate the environment:
   ```sh
   conda activate ecmwf_forecasts
   ```
5. Your environment is now ready to use with all required packages installed.

## Launching the Jupyter Notebook

After installing and activating your conda environment, you can launch desired notebook as follows:

1. Open Anaconda Prompt or your terminal.
2. Activate your environment:
   ```sh
   conda activate ecmwf_forecasts
   ```
3. Launch the notebook directly:
   ```sh
   jupyter notebook src/ecmwf_event_forecasting.ipynb
   ```

## Running the ECMWF_forecasting_raster.py script

You can run the main Python script directly from your shell. Make sure you are in the root folder of the project and have activated the required environment (see instructions above). Example:

```sh
cd path/to/grib_utils
conda activate ecmwf_forecasts
```

Then run the script with the following arguments:

### Mandatory arguments

- `dir_out` (str): Output directory where results will be saved.
- `format` (str): Output format, must be either `geotiff` or `netcdf`.

### Optional arguments

-- `--basedate` (str): Base date for the forecast in `YYYYMMDD` format. Default is today's date.
-- `--basetime` (int): Base time (hour, UTC) for the forecast. Default is `0` (midnight).
-- `--horizon` (int): Forecast horizon in hours (maximum lead time). Default is `240`.
-- `--timestep` (int): Forecast timestep in hours (interval between outputs). Default is `24`.
-- `--stack` (flag): If set, stack all timesteps into a single output file, else outputs into separate files for each timestep.

### Example usage

```sh
python src/ECMWF_forecasting_raster.py /path/to/output netcdf --basedate 20240824 --basetime 0 --horizon 240 --timestep 24 --stack
```

## Docker Usage

### Build and Run the Docker Container

1. **Build the Docker image**
   Open a terminal in the project root and run:
   ```sh
   docker build -t raster_forecast .
   ```

2. **Run the container**
   Run the container and pass arguments to the script inside Docker. For example:
   ```sh
   docker run --rm -v /path/to/output:/app/output raster_forecast /app/output netcdf --basedate 20240824 --basetime 0 --horizon 240 --timestep 24 --stack
   ```
   - Replace `/path/to/output` with the absolute path to your desired output directory on your host machine.
   - The arguments `/app/output netcdf --basedate 20250911 --basetime 0 --horizon 240 --timestep 24 --stack` correspond to the output directory, format, and all optional parameters.

3. **Override CMD or pass additional arguments**
   By default, Docker will use the CMD specified in the Dockerfile if you do not provide arguments after the image name. To run a different script or pass additional arguments, simply add them after the image name. For example, to run a different script or change parameters:
   ```sh
   docker run --rm -v /path/to/output:/app/output raster_forecast python src/ECMWF_forecasting_raster.py /app/output geotiff --basedate 20240824 --basetime 0 --horizon 240 --timestep 24 --stack
   ```
   This example explicitly calls the Python script and passes all arguments, overriding any default CMD in the Dockerfile.

This will execute the script inside the container with your specified options and write results to the mounted output directory.
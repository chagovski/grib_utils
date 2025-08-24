import os
import re
import csv
from datetime import datetime
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin

def get_grib_data(client_name, parameters, outpath, date=0, time=0, step=24, stream="oper", type_="fc", levtype="sfc"):
    """
    Retrieves GRIB data using the ECMWF Open Data Client.

    Args:
        client_name (str): The name of the client (e.g., "ecmwf").
        parameters (list): List of parameters to retrieve (e.g., ['tp']).
        outpath (str): Path to save the retrieved GRIB file.
        date (int): Date for the data retrieval (default: 0).
        time (int): Time for the data retrieval (default: 0).
        step (int): Step for the data retrieval (default: 24).
        stream (str): Stream type (default: "oper").
        type_ (str): Type of data (default: "fc").
        levtype (str): Level type (default: "sfc").
    """
    from ecmwf.opendata import Client

    client = Client(client_name, beta=False)
    client.retrieve(
        date=date,
        time=time,
        step=step,
        stream=stream,
        type=type_,
        levtype=levtype,
        param=parameters,
        target=outpath
    )

def grib_to_raster(filepath, outpath=None):
    """
    Converts a GRIB file containing ECMWF total precipitation data to a GeoTIFF raster.
    If multiple steps are present, saves each step as a separate band.
    """
    ds = xr.open_dataset(filepath, engine="cfgrib", decode_timedelta=True)
    tp = ds['tp']  # total precipitation variable

    # Convert from meters to millimeters
    tp_mm = tp * 1000
    tp_mm.attrs['units'] = 'mm'

    # Output path
    if outpath is None:
        tif_path = os.path.splitext(filepath)[0] + ".tif"
    else:
        tif_path = outpath

    # Get coordinates
    lon = tp_mm.longitude.values
    lat = tp_mm.latitude.values
    res_x = abs(lon[1] - lon[0])
    res_y = abs(lat[1] - lat[0])
    transform = from_origin(lon.min() - res_x/2, lat.max() + res_y/2, res_x, res_y)

    # Handle multiple steps (timesteps)
    data = tp_mm.values
    if data.ndim == 3:  # (step, y, x)
        bands = data.shape[0]
        height = data.shape[1]
        width = data.shape[2]
    elif data.ndim == 2:  # (y, x)
        bands = 1
        height = data.shape[0]
        width = data.shape[1]
        data = data[np.newaxis, :, :]  # add band dimension
    else:
        raise ValueError("Unexpected data shape: {}".format(data.shape))

    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        # Set band names for timesteps if available
        if "step" in tp_mm.dims:
            steps = ds["step"].values
            step_hours = [int(s / np.timedelta64(1, 'h')) for s in steps]
            band_names = [f"step_{h}h" for h in step_hours]
            for i in range(bands):
                dst.write(data[i, :, :], i + 1)
                dst.set_band_description(i + 1, band_names[i])
        else:
            dst.write(data[0, :, :], 1)

    print(f"GeoTIFF written to {tif_path}")

def store_metadata(
    raster_path, csv_path, variable
):

    filename = os.path.basename(raster_path)

    # Extract basetime, horizon, steplist from filename
    match = re.search(r'base(\d{8})T(\d{2})Z_h(\d+)_step(\d+)', filename)
    if not match:
        raise ValueError("Filename does not match expected format.")
    basetime = f"{match.group(1)}T{match.group(2)}"
    horizon = int(match.group(3))
    step = int(match.group(4))
    steplist = list(range(step, horizon + 1, step))

    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height
        epsg_code = src.crs.to_epsg() if src.crs else None
        crs = f"EPSG:{epsg_code}" if epsg_code else src.crs.to_string() if src.crs else ""
        bounds = src.bounds

        xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top

        # Calculate latitude and longitude steps and lists
        lat_step = (ymax - ymin) / height
        lon_step = (xmax - xmin) / width

        lats = [ymin + i * lat_step for i in range(height)]
        lons = [xmin + i * lon_step for i in range(width)]

        # Get current timestamp
        download_time = datetime.now().isoformat(timespec='seconds')

        row = [
            filename, variable, basetime, horizon, steplist, crs, lats, lons,
            width, height, xmin, ymin, xmax, ymax, download_time
        ]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "filename", "variable", "basetime", "horizon", "steps",
                "crs", "lats", "lons", "width", "height", "xmin", "ymin", "xmax", "ymax", "download_time"
            ])
        writer.writerow(row)

    print(f"Appended metadata for {filename} to {csv_path}")
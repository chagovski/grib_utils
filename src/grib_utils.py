import os
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
        for i in range(bands):
            dst.write(data[i, :, :], i + 1)

    print(f"GeoTIFF written to {tif_path}")
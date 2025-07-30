import os
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

    Args:
        filepath (str): Path to the input GRIB file.
        outpath (str, optional): Path to the output GeoTIFF file. If None, saves as
            filepath with '.tif' extension.

    The function reads the GRIB file, extracts the 'tp' (total precipitation) variable,
    converts its units from meters to millimeters, and writes the data as a single-band
    GeoTIFF with geographic coordinates (EPSG:4326).

    The output GeoTIFF is saved in the same directory as the input file, with the
    extension changed to '.tif' if outpath is not specified.

    Example:
        grib_to_raster("/path/to/file.grib")
        grib_to_raster("/path/to/file.grib", "/path/to/output.tif")
    """

    ds = xr.open_dataset(filepath, engine="cfgrib", decode_timedelta=True)
    tp = ds['tp']  # total precipitation variable

    # Convert from meters to millimeters
    tp_mm = tp * 1000
    tp_mm.attrs['units'] = 'mm'

    # Since time does not exist, use tp_mm directly
    data = tp_mm.values.astype('float32')

    # Get coordinates
    lon = tp_mm.longitude.values
    lat = tp_mm.latitude.values

    # Calculate resolution
    res_x = abs(lon[1] - lon[0])
    res_y = abs(lat[1] - lat[0])

    # Build transform (assumes north-up, regular grid)
    transform = from_origin(lon.min() - res_x/2, lat.max() + res_y/2, res_x, res_y)

    # Output path
    if outpath is None:
        tif_path = os.path.splitext(filepath)[0] + ".tif"
    else:
        tif_path = outpath

    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    print(f"GeoTIFF written to {tif_path}")
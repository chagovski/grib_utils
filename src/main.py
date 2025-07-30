# Example for main.py
from grib_utils import get_grib_data, grib_to_raster

if __name__ == "__main__":

    out_path = "path/to/sample.grb"

    # Download GRIB file
    get_grib_data(
        client_name="ecmwf",
        parameters=["tp"],
        outpath=out_path,
        date=20250730,  # format: YYYYMMDD
        time=0,
        step=24,
        stream="oper",
        type_="fc",
        levtype="sfc"
    )

    # Convert GRIB to GeoTIFF
    grib_to_raster(out_path)
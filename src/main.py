# Example for main.py
import os
from datetime import datetime
from grib_utils import get_grib_data, grib_to_raster, store_metadata

if __name__ == "__main__":

    output_dir = "path/to/output/dir"
    metadata_path = os.path.join(output_dir, "ecmwf_forecasts.csv")

    basedate = datetime.today().strftime('%Y%m%d')
    basetime = 00 
    horizon = 240
    timestep = 24
    filename = f"ECMWF_total_accumulated_precipitation_forecast_base{basedate}T{basetime:02d}Z_h{horizon}_step{timestep}.grb"

    output_path = os.path.join(output_dir, filename)


    # Download GRIB file
    get_grib_data(
        client_name="ecmwf",
        parameters=["tp"],
        outpath=output_path,
        date=basedate,  # format: YYYYMMDD
        time=basetime, # format: HH (00 for midnight, 12 for noon)
        step=list(range(timestep, horizon + 1, timestep)), # 0 to 144 by 3, 150 to 240 by 6
        stream="oper",
        type_="fc",
        levtype="sfc"
    )


    # Convert GRIB to GeoTIFF
    grib_to_raster(output_path)

    store_metadata(
        raster_path=output_path.replace(".grb", ".tif"),
        csv_path=metadata_path,
        variable="total_precipitation"
    )
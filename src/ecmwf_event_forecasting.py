
import os
import argparse
from datetime import datetime
from grib_utils import get_grib_data, grib_to_netcdf

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download and convert ECMWF precipitation forecast.")
    parser.add_argument("dir_out", type=str, help="Output directory (mandatory)")
    parser.add_argument("--basedate", type=str, default=datetime.today().strftime('%Y%m%d'), help="Base date (YYYYMMDD), default: today")
    parser.add_argument("--basetime", type=int, default=0, help="Base time (HH), default: 00")
    parser.add_argument("--horizon", type=int, default=240, help="Forecast horizon in hours, default: 240")
    parser.add_argument("--timestep", type=int, default=24, help="Forecast timestep in hours, default: 24")
    args = parser.parse_args()

    output_dir = args.dir_out
    basedate = args.basedate
    basetime = args.basetime
    horizon = args.horizon
    timestep = args.timestep

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

    # Convert GRIB to netcdf
    grib_to_netcdf(output_path, stack=True)
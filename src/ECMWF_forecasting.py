import os
import argparse
from datetime import datetime
from grib_utils import get_grib_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download and convert ECMWF precipitation forecast.")
    parser.add_argument("dir_out", type=str, help="Output directory (mandatory)")
    parser.add_argument("format", type=str, choices=["geotiff", "netcdf", "geopackage"], help="Output format: 'geotiff', 'netcdf', or 'geopackage' (mandatory)")
    parser.add_argument("--basedate", type=str, default=datetime.today().strftime('%Y%m%d'), help="Base date (YYYYMMDD), default: today")
    parser.add_argument("--basetime", type=int, default=0, help="Base time (HH), default: 00")
    parser.add_argument("--horizon", type=int, default=240, help="Forecast horizon in hours, default: 240")
    parser.add_argument("--timestep", type=int, default=24, help="Forecast timestep in hours, default: 24")
    parser.add_argument("--stack", action="store_true", help="Stack all timesteps into a single output file. By default, outputs will be split by timestep unless this flag is set.")

    # GeoPackage specific arguments
    gpkg_group = parser.add_argument_group('GeoPackage Conversion Options')
    gpkg_group.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        help='List of precipitation thresholds (mm) for vectorization. Mutually exclusive with --mode/--n_classes.'
    )
    gpkg_group.add_argument(
        '--mode',
        choices=['jenks', 'equal'],
        default=None,
        help='Classification mode for auto-thresholding: "jenks" or "equal". Requires --n_classes.'
    )
    gpkg_group.add_argument(
        '--n_classes',
        type=int,
        default=None,
        help='Number of classes for auto-thresholding. Requires --mode.'
    )
    gpkg_group.add_argument(
        '--vector_type',
        choices=['contour', 'polygon'],
        default='contour',
        help="Type of vector output for geopackage: 'contour' or 'polygon'"
    )
    gpkg_group.add_argument(
        '--flatten',
        action='store_true',
        help="Merge all features for each threshold into a single MultiLineString (contour) or MultiPolygon (polygon) feature. Default: False."
    )
    
    args = parser.parse_args()

    output_dir = args.dir_out
    out_format = args.format.lower()
    basedate = args.basedate
    basetime = args.basetime
    horizon = args.horizon
    timestep = args.timestep
    stack = args.stack

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

    # Convert GRIB to requested format
    if out_format == "netcdf":
        from grib_utils import grib_to_netcdf
        grib_to_netcdf(output_path, stack=stack)
    elif out_format == "geotiff":
        from grib_utils import grib_to_geotiff
        grib_to_geotiff(output_path, stack=stack)
    elif out_format == "geopackage":
        from grib_utils import grib_to_geopackage
        grib_to_geopackage(
            output_path,
            thresholds=args.thresholds,
            mode=args.mode,
            n_classes=args.n_classes,
            stack=stack,
            flatten=args.flatten,
            vector_type=args.vector_type
        )
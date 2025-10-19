#!/usr/bin/env python3
"""
Script to download ECMWF forecast data and convert it to GeoPackage format.
"""

import os
import numpy as np
from grib_utils import get_grib_data, grib_to_geopackage

def main():
    # Configuration
    client_name = "ecmwf"
    parameters = ['tp']  # total precipitation
    
    # Output directory and file paths
    output_dir = r"S:\TMP\AP\temp"
    os.makedirs(output_dir, exist_ok=True)
    
    grib_file = os.path.join(output_dir, "ecmwf_forecast.grib")
    gpkg_file = os.path.join(output_dir, "ecmwf_forecast.gpkg")
    
    print("Starting ECMWF forecast download and conversion...")
    
    # Step 1: Download GRIB data
    print(f"1. Downloading ECMWF forecast data to {grib_file}")
    try:
        get_grib_data(
            client_name=client_name,
            parameters=parameters,
            outpath=grib_file,
            date=0,  # Latest available date
            time=0,  # 00Z run
            step=[240],  # Every 24 hours up to 240 hours (10 days)
            stream="oper",  # Operational forecast
            type_="fc",  # Forecast
            levtype="sfc"  # Surface level
        )
        print("✓ Download completed successfully")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return
    
    # Step 2: Convert to GeoPackage
    print(f"2. Converting GRIB to GeoPackage: {gpkg_file}")
    try:
        grib_to_geopackage(
            filepath=grib_file,
            outpath=gpkg_file,
            mode="jenks",
            n_classes=9,
            # stack=True  # All timesteps in one file with multiple layers
        )
        print("✓ Conversion to GeoPackage completed successfully")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return
    
    print(f"📁 GRIB file: {grib_file}")
    print(f"📁 GeoPackage file: {gpkg_file}")

if __name__ == "__main__":
    main()

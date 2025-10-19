import os
import re
import csv
from datetime import datetime
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

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


def grib_to_geotiff(filepath, outpath=None, stack=True):
    """
    Converts a GRIB file containing ECMWF total precipitation data to a GeoTIFF raster.
    If multiple steps are present, saves each step as a separate band (default) or as separate files if separate_timesteps=True.
    """
    ds = xr.open_dataset(filepath, engine="cfgrib", decode_timedelta=True)
    tp = ds['tp']  # total precipitation variable

    # Convert from meters to millimeters
    tp_mm = tp * 1000
    tp_mm.attrs['units'] = 'mm'

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
        raise ValueError(f"Unexpected data shape: {data.shape}")

    if not stack and bands > 1:
        # Save each timestep as a separate GeoTIFF file
        if outpath is None:
            base_path = os.path.splitext(filepath)[0]
        else:
            base_path = os.path.splitext(outpath)[0]
        steps = ds["step"].values if "step" in tp_mm.dims else range(bands)
        step_hours = [int(s / np.timedelta64(1, 'h')) if isinstance(s, np.timedelta64) else int(s) for s in steps]
        for i in range(bands):
            # Extract base/horizon/step from original filename
            orig_filename = os.path.basename(filepath)
            match = re.search(r'base(\d{8})T(\d{2})Z_h(\d+)_step(\d+)', orig_filename)
            if match:
                basetime = f"base{match.group(1)}T{match.group(2)}Z"
                # For individual layer, horizon and step are both step_hours[i]
                horizon = step_hours[i]
                step_val = step_hours[i]
                tif_filename = f"ECMWF_total_accumulated_precipitation_forecast_{basetime}_h{horizon}_step{step_val}.tif"
                tif_path = os.path.join(os.path.dirname(filepath), tif_filename)
            else:
                tif_path = f"{base_path}_step_{step_hours[i]}h.tif"
            with rasterio.open(
                tif_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype='float32',
                crs='EPSG:4326',
                transform=transform,
            ) as dst:
                dst.write(data[i, :, :], 1)
            print(f"GeoTIFF written to {tif_path}")
    else:
        # Save all timesteps as bands in one file
        if outpath is None:
            out_path = os.path.splitext(filepath)[0] + ".tif"
        else:
            out_path = outpath
        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            if "step" in tp_mm.dims:
                steps = ds["step"].values
                step_hours = [int(s / np.timedelta64(1, 'h')) for s in steps]
                band_names = [f"step_{h}h" for h in step_hours]
                for i in range(bands):
                    dst.write(data[i, :, :], i + 1)
                    dst.set_band_description(i + 1, band_names[i])
            else:
                dst.write(data[0, :, :], 1)
        print(f"GeoTIFF written to {out_path}")

def grib_to_netcdf(filepath, outpath=None, stack=True):
    """
    Converts a GRIB file containing ECMWF total precipitation data to NetCDF format.
    Adds CF-compliant CRS variable for QGIS compatibility.
    If separate_timesteps=True, saves each timestep as a separate NetCDF file.
    """
    ds = xr.open_dataset(filepath, engine="cfgrib", decode_timedelta=True)
    tp = ds['tp']  # total precipitation variable

    # Convert from meters to millimeters
    tp_mm = tp * 1000
    tp_mm.attrs['units'] = 'mm'

    data = tp_mm.values
    if data.ndim == 3:  # (step, y, x)
        bands = data.shape[0]
    else:
        bands = 1

    wgs84_wkt = (
        'GEOGCRS["WGS 84",'
        'DATUM["World Geodetic System 1984",'
        'ELLIPSOID["WGS 84",6378137,298.257223563,'
        'LENGTHUNIT["metre",1.0]]],'
        'PRIMEM["Greenwich",0.0,ANGLEUNIT["degree",0.0174532925199433]],'
        'CS[ellipsoidal,2],'
        'AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],'
        'AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],'
        'ID["EPSG",4326]]'
    )
    if not stack and bands > 1:
        # Save each timestep as a separate NetCDF file
        steps = ds["step"].values if "step" in tp_mm.dims else range(bands)
        step_hours = [int(s / np.timedelta64(1, 'h')) if isinstance(s, np.timedelta64) else int(s) for s in steps]
        for i in range(bands):
            orig_filename = os.path.basename(filepath)
            match = re.search(r'base(\d{8})T(\d{2})Z_h(\d+)_step(\d+)', orig_filename)
            if match:
                basetime = f"base{match.group(1)}T{match.group(2)}Z"
                horizon = step_hours[i]
                step_val = step_hours[i]
                nc_filename = f"ECMWF_total_accumulated_precipitation_forecast_{basetime}_h{horizon}_step{step_val}.nc"
                nc_path = os.path.join(os.path.dirname(filepath), nc_filename)
            else:
                nc_path = f"{os.path.splitext(filepath)[0]}_step_{step_hours[i]}h.nc"
            single_tp = xr.DataArray(
                data[i, :, :],
                dims=("latitude", "longitude"),
                coords={
                    "latitude": tp_mm.latitude,
                    "longitude": tp_mm.longitude
                },
                name="tp_mm",
                attrs=tp_mm.attrs
            )
            ds_mm = single_tp.to_dataset()
            # Remove 'surface' coordinate if present
            if 'surface' in ds_mm.coords:
                ds_mm = ds_mm.drop_vars('surface')
            # Rename 'time' coordinate to 'basetime' if present
            if 'time' in ds_mm.coords:
                ds_mm = ds_mm.rename({'time': 'basetime'})
            # Swap main dimension to 'valid_time' if possible
            if 'step' in ds_mm.dims and 'valid_time' in ds_mm.coords:
                ds_mm = ds_mm.swap_dims({'step': 'valid_time'})
            # Robustly rename 'valid_time' to 'std_time' after swapping
            if 'valid_time' in ds_mm.dims:
                ds_mm = ds_mm.rename({'valid_time': 'std_time'})

            # Only copy global attributes that do not start with 'GRIB_'
            filtered_attrs = {k: v for k, v in ds.attrs.items() if not k.startswith("GRIB_")}
            ds_mm.attrs.update(filtered_attrs)
            # Add CF-compliant CRS variable with WKT
            crs_var = xr.DataArray(
                0,
                attrs={
                    "grid_mapping_name": "latitude_longitude",
                    "spatial_ref": wgs84_wkt,
                    "epsg_code": "EPSG:4326",
                    "semi_major_axis": 6378137.0,
                    "inverse_flattening": 298.257223563,
                },
            )
            ds_mm["crs"] = crs_var
            ds_mm["tp_mm"].attrs["grid_mapping"] = "crs"
            ds_mm.attrs["crs"] = "EPSG:4326"
            ds_mm.attrs["spatial_ref"] = wgs84_wkt
            ds_mm.to_netcdf(nc_path)
            print(f"NetCDF written to {nc_path}")
    else:
        # Save all timesteps in one NetCDF file
        if outpath is None:
            out_path = os.path.splitext(filepath)[0] + ".nc"
        else:
            out_path = outpath
        ds_mm = tp_mm.to_dataset(name="tp_mm")
        # Remove 'surface' coordinate if present
        if 'surface' in ds_mm.coords:
            ds_mm = ds_mm.drop_vars('surface')
        # Rename 'time' coordinate to 'basetime' if present
        if 'time' in ds_mm.coords:
            ds_mm = ds_mm.rename({'time': 'basetime'})
        # Swap main dimension to 'valid_time' if possible
        if 'step' in ds_mm.dims and 'valid_time' in ds_mm.coords:
            ds_mm = ds_mm.swap_dims({'step': 'valid_time'})
        # Robustly rename 'valid_time' to 'std_time' after swapping
        if 'valid_time' in ds_mm.dims:
            ds_mm = ds_mm.rename({'valid_time': 'std_time'})

        # Only copy global attributes that do not start with 'GRIB_'
        filtered_attrs = {k: v for k, v in ds.attrs.items() if not k.startswith("GRIB_")}
        ds_mm.attrs.update(filtered_attrs)
        # Add CF-compliant CRS variable with WKT
        crs_var = xr.DataArray(
            0,
            attrs={
                "grid_mapping_name": "latitude_longitude",
                "spatial_ref": wgs84_wkt,
                "epsg_code": "EPSG:4326",
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
            },
        )
        ds_mm["crs"] = crs_var
        ds_mm["tp_mm"].attrs["grid_mapping"] = "crs"
        ds_mm.attrs["crs"] = "EPSG:4326"
        ds_mm.attrs["spatial_ref"] = wgs84_wkt
        ds_mm.to_netcdf(out_path)
        print(f"NetCDF written to {out_path}")

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
    # If horizon == step, it's a single-layer file, so steplist should be [step]
    if horizon == step:
        steplist = [step]
    else:
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

def grib_to_geopackage(filepath, outpath=None, thresholds=None, stack=True, mode=None, n_classes=None):
    """
    Converts a GRIB file containing ECMWF total precipitation data to GeoPackage format.
    Vectorizes the raster data based on provided thresholds.
    
    Args:
        filepath (str): Path to the input GRIB file.
        outpath (str, optional): Path to save the output GeoPackage. If None, uses input filename with .gpkg extension.
        thresholds (list, optional): List of threshold values for vectorization. 
                                   Mutually exclusive with mode/n_classes.
        stack (bool): If True, processes all timesteps together. If False, creates separate files for each timestep.
        mode (str, optional): Classification method - "jenks" for natural breaks or "equal" for equal intervals.
                             Must be used together with n_classes. Mutually exclusive with thresholds.
        n_classes (int, optional): Number of classes for classification. Must be used together with mode.
                                 Mutually exclusive with thresholds.
    
    Returns:
        None
    """
    # Load GRIB data
    ds = xr.open_dataset(filepath, engine="cfgrib", decode_timedelta=True)
    tp = ds['tp']  # total precipitation variable

    # Convert from meters to millimeters
    tp_mm = tp * 1000
    tp_mm.attrs['units'] = 'mm'
    
    # Parameter validation - ensure mutual exclusivity
    if thresholds is not None and (mode is not None or n_classes is not None):
        raise ValueError("thresholds parameter is mutually exclusive with mode/n_classes parameters")
    
    if (mode is not None and n_classes is None) or (mode is None and n_classes is not None):
        raise ValueError("mode and n_classes must both be specified together or both be None")
    
    if mode is not None and mode not in ["jenks", "equal"]:
        raise ValueError("mode must be either 'jenks' or 'equal'")
    
    # Generate thresholds based on parameters
    if thresholds is not None:
        # Use custom thresholds
        print(f"Using custom thresholds: {thresholds}")
    elif mode is not None and n_classes is not None:
        # Use classification method
        print(f"Using {mode} classification with {n_classes} classes")
        
        # Get all precipitation values (excluding zeros for better classification)
        all_values = tp_mm.values.flatten()
        precip_values = all_values[all_values > 0]
        
        if len(precip_values) == 0:
            print("No precipitation data found, using minimal thresholds")
            thresholds = [0.1, 1, 5]
        else:
            max_precip = float(np.nanmax(precip_values))
            print(f"Maximum precipitation in data: {max_precip:.2f} mm")
            
            if mode == "equal":
                # Equal interval classification
                min_precip = 0  # Start from 0
                interval = max_precip / n_classes
                thresholds = [interval * (i + 1) for i in range(n_classes)]
                print(f"Equal interval thresholds: {[round(t, 2) for t in thresholds]}")
                
            elif mode == "jenks":
                # Natural breaks (Jenks) classification
                try:
                    import jenkspy
                    # Sample data if too large for performance
                    if len(precip_values) > 10000:
                        sample_size = 10000
                        precip_sample = np.random.choice(precip_values, sample_size, replace=False)
                    else:
                        precip_sample = precip_values
                    
                    # Get jenks breaks
                    breaks = jenkspy.jenks_breaks(precip_sample, n_classes)
                    # Remove the first break (minimum value) to get thresholds
                    thresholds = breaks[1:]
                    print(f"Jenks natural breaks thresholds: {[round(t, 2) for t in thresholds]}")
                    
                except ImportError:
                    print("Warning: jenkspy not available, falling back to equal intervals")
                    # Fallback to equal intervals
                    min_precip = 0
                    interval = max_precip / n_classes
                    thresholds = [interval * (i + 1) for i in range(n_classes)]
                    print(f"Equal interval thresholds (fallback): {[round(t, 2) for t in thresholds]}")
    else:
        # Default strategy: every 50mm up to max value
        max_precip = float(np.nanmax(tp_mm.values))
        print(f"Maximum precipitation in data: {max_precip:.2f} mm")
        
        if max_precip <= 0:
            print("No precipitation data found, using minimal thresholds")
            thresholds = [0.1, 1, 5]
        else:
            # Calculate the upper bound that includes max_precip
            # Round up to next 50mm increment
            upper_bound = int(np.ceil(max_precip / 50) * 50)
            
            # Create thresholds every 50mm from 50 to upper_bound
            thresholds = list(range(50, upper_bound + 1, 50))
            
            print(f"Auto-generated thresholds every 50mm up to {upper_bound}mm: {thresholds}")

    # Get coordinates and transformation
    lon = tp_mm.longitude.values
    lat = tp_mm.latitude.values
    res_x = abs(lon[1] - lon[0])
    res_y = abs(lat[1] - lat[0])
    transform = from_origin(lon.min() - res_x/2, lat.max() + res_y/2, res_x, res_y)

    # Handle data dimensions
    data = tp_mm.values
    if data.ndim == 3:  # (step, y, x)
        bands = data.shape[0]
    elif data.ndim == 2:  # (y, x)
        bands = 1
        data = data[np.newaxis, :, :]  # add band dimension
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    def vectorize_raster_with_thresholds(raster_data, transform, thresholds):
        """
        Vectorize raster data based on threshold ranges and return GeoDataFrame.
        """
        geometries = []
        
        # Get actual maximum value from the raster data
        actual_max = float(np.nanmax(raster_data))
        
        # Sort thresholds to ensure proper ordering
        sorted_thresholds = sorted(thresholds)
        
        # Create threshold ranges
        ranges = []
        for i in range(len(sorted_thresholds)):
            if i == 0:
                # First range: 0 to first threshold
                ranges.append((0, sorted_thresholds[i]))
            # Range from current threshold to next threshold
            if i < len(sorted_thresholds) - 1:
                ranges.append((sorted_thresholds[i], sorted_thresholds[i + 1]))
        
        # Last range: last threshold to actual maximum value
        ranges.append((sorted_thresholds[-1], actual_max))
        
        for min_val, max_val in ranges:
            # Create mask for current threshold range
            if min_val == max_val:
                # Skip if range has no width
                continue
            elif max_val == actual_max and min_val < actual_max:
                # Last range includes the maximum value
                mask = (raster_data >= min_val).astype(np.uint8)
                range_label = f"{min_val} - {max_val:.2f} mm"
                max_threshold_val = max_val
            else:
                mask = ((raster_data >= min_val) & (raster_data < max_val)).astype(np.uint8)
                range_label = f"{min_val} - {max_val} mm"
                max_threshold_val = max_val
            
            # Skip if no pixels in this range
            if not np.any(mask):
                continue
            
            # Convert raster to polygons
            for geom, value in shapes(mask, mask=mask, transform=transform):
                if value == 1:  # Only process areas within the threshold
                    geometries.append({
                        'geometry': shape(geom),
                        'threshold_range': range_label,
                        'min_threshold': min_val,
                        'max_threshold': max_threshold_val,
                        'precipitation_mm': range_label
                    })
        
        return gpd.GeoDataFrame(geometries, crs='EPSG:4326')

    if not stack and bands > 1:
        # Save each timestep as a separate GeoPackage file
        steps = ds["step"].values if "step" in tp_mm.dims else range(bands)
        step_hours = [int(s / np.timedelta64(1, 'h')) if isinstance(s, np.timedelta64) else int(s) for s in steps]
        
        for i in range(bands):
            # Generate output filename for individual timestep
            orig_filename = os.path.basename(filepath)
            match = re.search(r'base(\d{8})T(\d{2})Z_h(\d+)_step(\d+)', orig_filename)
            if match:
                basetime = f"base{match.group(1)}T{match.group(2)}Z"
                horizon = step_hours[i]
                step_val = step_hours[i]
                gpkg_filename = f"ECMWF_total_accumulated_precipitation_forecast_{basetime}_h{horizon}_step{step_val}.gpkg"
                gpkg_path = os.path.join(os.path.dirname(filepath), gpkg_filename)
            else:
                gpkg_path = f"{os.path.splitext(filepath)[0]}_step_{step_hours[i]}h.gpkg"
            
            # Vectorize current timestep
            gdf = vectorize_raster_with_thresholds(data[i, :, :], transform, thresholds)
            
            if len(gdf) > 0:
                # Add timestep metadata
                gdf['timestep_hours'] = step_hours[i]
                gdf['basetime'] = match.group(1) + 'T' + match.group(2) if match else 'unknown'
                
                # Save to GeoPackage
                gdf.to_file(gpkg_path, driver='GPKG')
                print(f"GeoPackage written to {gpkg_path} with {len(gdf)} features")
            else:
                print(f"No features generated for timestep {step_hours[i]}h - skipping file creation")
    
    else:
        # Save all timesteps in one GeoPackage file with different layers
        if outpath is None:
            out_path = os.path.splitext(filepath)[0] + ".gpkg"
        else:
            out_path = outpath
        
        if bands > 1:
            steps = ds["step"].values if "step" in tp_mm.dims else range(bands)
            step_hours = [int(s / np.timedelta64(1, 'h')) if isinstance(s, np.timedelta64) else int(s) for s in steps]
            
            # Process each timestep as a separate layer
            for i in range(bands):
                gdf = vectorize_raster_with_thresholds(data[i, :, :], transform, thresholds)
                
                if len(gdf) > 0:
                    # Add timestep metadata
                    gdf['timestep_hours'] = step_hours[i]
                    layer_name = f"precipitation_step_{step_hours[i]}h"
                    
                    # Save as layer in GeoPackage
                    if i == 0:
                        gdf.to_file(out_path, driver='GPKG', layer=layer_name)
                    else:
                        gdf.to_file(out_path, driver='GPKG', layer=layer_name, mode='a')
                    
                    print(f"Layer '{layer_name}' written to {out_path} with {len(gdf)} features")
                else:
                    print(f"No features generated for timestep {step_hours[i]}h - skipping layer")
        else:
            # Single timestep
            gdf = vectorize_raster_with_thresholds(data[0, :, :], transform, thresholds)
            
            if len(gdf) > 0:
                gdf.to_file(out_path, driver='GPKG')
                print(f"GeoPackage written to {out_path} with {len(gdf)} features")
            else:
                print(f"No features generated - skipping file creation")
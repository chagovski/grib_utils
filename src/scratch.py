import xarray as xr

ds = xr.open_dataset("/mnt/company_general/TMP/AP/temp/ECMWF_total_accumulated_precipitation_forecast_base20251012T00Z_h240_step24.nc")
print(ds)  # Shows all variables, coordinates, and global attributes

# To list all variable names:
print(ds.data_vars)

# To list all global attributes:
print(ds.attrs)


# Print the contents of the 'valid_time' dimension (if present)
# print("valid_time dimension:", ds['valid_time'].values)





# Print dtypes of all variables
print("\nVariable dtypes:")
for var in ds.data_vars:
	print(f"{var}: {ds[var].dtype}")

# Print dtypes of all coordinates
print("\nCoordinate dtypes:")
for coord in ds.coords:
	print(f"{coord}: {ds.coords[coord].dtype}")

# Print dtypes of all dimensions
print("\nDimension dtypes:")
for dim in ds.dims:
	arr = ds[dim] if dim in ds.data_vars else ds.coords.get(dim, None)
	if arr is not None:
		print(f"{dim}: {arr.dtype}")

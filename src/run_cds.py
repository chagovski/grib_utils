from CDS_utils import get_era5_tp
from grib_utils import grib_to_geotiff

# Download ERA5 GRIB file
grib_path = 'C:\\Users\\pasik\\private\\era5_precip.grib'
get_era5_tp(
    year='2025',
    month='08',
    days=['01'],
    times=['00:00'],
    out_path=grib_path
)

# Convert GRIB to GeoTIFF
geotiff_path = grib_path.replace('.grib', '.tif')
grib_to_geotiff(grib_path, outpath=geotiff_path)
print(f"GeoTIFF written to {geotiff_path}")
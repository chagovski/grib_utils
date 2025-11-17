import cdsapi

def get_era5_tp(year, month, days, times, out_path, area=None):
    """
    Download global ERA5 total precipitation data from Copernicus Climate Data Store (CDS).
    Args:
        year (str or int): Year, e.g. '2025'
        month (str or int): Month, e.g. '11'
        days (list of str): List of days, e.g. ['01', '02', '03']
        times (list of str): List of times, e.g. ['00:00', '06:00', '12:00', '18:00']
        out_path (str): Output NetCDF file path
        area (list, optional): [N, W, S, E] bounding box. If None, downloads global data.
    """
    c = cdsapi.Client()
    request = {
        'variable': 'total_precipitation',
        'year': str(year),
        'month': str(month),
        'day': days,
        'time': times,
        'format': 'grib',
    }
    if area:
        request['area'] = area
    c.retrieve(
        'reanalysis-era5-single-levels',
        request,
        out_path
    )
    print(f"Downloaded ERA5 precipitation to {out_path}")

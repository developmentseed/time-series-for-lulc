import glob
from pathlib import Path

import geopandas as gpd
import pystac_client
import stackstac
import xarray
from dask.distributed import Client, LocalCluster
from numcodecs import Zstd
from rasterio.features import rasterize

CLASS_DN_LOOKUP = {
    "agriculture": 1,
    "bare_soil": 2,
    "dry_jungle": 3,
    "forest": 4,
    "humid_jungle": 5,
    "pasture": 6,
    "scrub": 7,
    "urban": 8,
    "water": 9,
    "without_apparent_vegetation": 10,
}

wd = "/home/tam/Documents/devseed"

cluster = LocalCluster()  # Launches a scheduler and workers locally
client = Client(cluster)
catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v0/")

epsg = 6362

for geojson in glob.glob(f"{wd}/geojson_sentinel/*.geojson"):
    print("Working on file", geojson)
    src = gpd.read_file(geojson)

    # Project geometries into mexican projection https://epsg.io/6362
    src_mx = src.to_crs(f"EPSG:{epsg}")

    # Search for imagery that intersects with the bbox of the file. This
    # bbox is required to be in lat/lon, so we use the original bounds.
    search = catalog.search(
        collections=["sentinel-s2-l2a-cogs"],
        bbox=src.total_bounds,
        datetime="2021-11-01/2022-04-30",
    )
    items = search.get_all_items()
    print(f"Found {len(items)} items")

    # Stack imagery in mexican bounds at 10m resolution.
    stack = stackstac.stack(
        items,
        bounds=src_mx.total_bounds,
        epsg=epsg,
        resolution=10,
        dtype="uint16",
        fill_value=0,
    )

    # Keep all bands with 10m or 20m resolution.
    data = stack.sel(
        band=[
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
            "SCL",
        ]
    )

    # Fetch data.
    data = data.compute()

    # Ensure data is writable to zarr
    data.attrs["transform"] = tuple(data.transform)
    del data.attrs["spec"]

    # Rasterize training data
    rasterized = rasterize(
        [
            (dat.geometry, CLASS_DN_LOOKUP[dat["class"]])
            for fid, dat in src_mx.iterrows()
        ],
        out_shape=data.shape[2:],
        transform=data.transform,
        all_touched=True,
        fill=0,
        dtype="uint8",
    )

    # Combine y and X array for training later
    combo = xarray.Dataset(
        {
            "imagery": data,
            "training": (["y", "x"], rasterized),
        }
    )

    # Save to zarr
    filepath = f"{wd}/stacks/{Path(geojson).stem}.zarr"
    print(f"Finished fetching {geojson}, writing to {filepath}")
    encoding = {dat: {"compressor": Zstd(level=9)} for dat in combo.data_vars}
    combo.to_zarr(filepath, mode="w", encoding=encoding)

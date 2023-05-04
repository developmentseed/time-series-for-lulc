import json

import cv2
import morecantile
import numpy
import pystac_client
import stackstac
import xarray
from rasterio.enums import Resampling
from rasterio.features import bounds as geomBounds
from rasterio.warp import transform_geom

# 0 NO_DATA
# 1 SATURATED_OR_DEFECTIVE
# 2 DARK_AREA_PIXELS
# 3 CLOUD_SHADOWS
# 4 VEGETATION
# 5 NOT_VEGETATED
# 6 WATER
# 7 UNCLASSIFIED
# 8 CLOUD_MEDIUM_PROBABILITY
# 9 CLOUD_HIGH_PROBABILITY
# 10 THIN_CIRRUS
# 11 SNOW
CLOUDY_OR_NODATA = (0, 3, 8, 9, 10)


catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v0/")
collection = "sentinel-s2-l2a-cogs"

# catalog = pystac_client.Client.open("https://landsatlook.usgs.gov/stac-server/")
# collection = "landsat-c2l2-sr"

epsg = 3857
zoom = 12

# Cretas, Greece
coordx, coordy = 35.580521, 23.592116
coordx, coordy = 23.47799, 38.18146
# Sahara
coordx, coordy = 7.89839, 26.90210
# Australia
# coordx, coordy = 121.41734, -32.81854

# Portugal
coordx, coordy = -9.15032, 38.72595

size = 256

tms = morecantile.tms.get("WebMercatorQuad")

first_tile = tms.tile(coordx, coordy, zoom)
print(first_tile)

tiles = []
for i in range(2):
    for j in range(2):
        tiles.append(morecantile.Tile(first_tile.x + i, first_tile.y + j, first_tile.z))


for tile in tiles:
    # start = "2017-06-01"
    start = "2022-01-01"
    search = catalog.search(
        collections=[collection],
        bbox=tms.bounds(tile),
        datetime=f"{start}/2023-04-30",
    )
    items = search.get_all_items()
    print(f"Found {len(items)} items")

    # Stack imagery
    stack = stackstac.stack(
        items,
        bounds=tms.xy_bounds(tile),
        snap_bounds=False,
        epsg=epsg,
        resolution=tms._resolution(tms.matrix(zoom=zoom)),
        dtype="uint16",
        fill_value=0,
        resampling=Resampling.bilinear,
    )

    # RGB as BGR plus scene class for cloud masking
    stack = stack.sel(
        band=[
            "B02",
            "B03",
            "B04",
            "SCL",
        ]
    )

    # Fetch data.
    data = stack.compute()

    scl = data.sel(band="SCL").astype("uint8")

    cloud_mask = scl.isin(CLOUDY_OR_NODATA)

    cleaned_data = data.where(~cloud_mask)

    # Create composites with cloud mask
    composites_using_cloud_mask = cleaned_data.resample(
        time="14D", skipna=True, origin=start, closed="right"
    ).median("time")

    # Create composites without cloud mask
    composites_using_all_pixels = data.resample(
        time="14D", skipna=True, origin=start, closed="right"
    ).median("time")

    # Fill pixels in cloud masked composites with pixels from full composite
    composites = xarray.where(
        numpy.isnan(composites_using_cloud_mask),
        composites_using_all_pixels,
        composites_using_cloud_mask,
    )

    filepath = f"/home/tam/Desktop/videomap/videomap-{tile.z}-{tile.x}-{tile.y}.mp4"

    out = cv2.VideoWriter(
        filepath, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=4, frameSize=(size, size)
    )
    for img in composites:
        bgr = img.data[:3].transpose(1, 2, 0)
        bgr = 255 * bgr / 3000
        bgr = numpy.clip(bgr, 0, 255).astype("uint8")
        out.write(bgr)

    geojson = f"/home/tam/Desktop/videomap/videomap-{tile.z}-{tile.x}-{tile.y}.geojson"

    with open(geojson, "w") as f:
        json.dump(tms.feature(tile), f)

    out.release()

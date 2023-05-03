import cv2
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
coordx, coordy = -985632, 4726291
resolution = 60
size = 256


def gen_bounds(coordx, coordy, resolution, size):
    step = resolution * size
    for i in [0, 1]:
        for j in [0, 1]:
            yield {
                "type": "Polygon",
                "coordinates": [
                    [
                        [coordx + i * step, coordy + j * step],
                        [coordx + (i + 1) * step, coordy + j * step],
                        [coordx + (i + 1) * step, coordy - (j + 1) * step],
                        [coordx + i * step, coordy - (j + 1) * step],
                        [coordx + i * step, coordy + j * step],
                    ]
                ],
            }


for geom in gen_bounds(coordx, coordy, resolution, size):
    # Search for imagery that intersects with the bbox of the file. This
    # bbox is required to be in lat/lon, so we use the original bounds.
    # start = "2017-06-01"
    start = "2023-01-01"
    search = catalog.search(
        collections=[collection],
        intersects=transform_geom(f"EPSG:{epsg}", "EPSG:4326", geom),
        datetime=f"{start}/2023-04-30",
    )
    items = search.get_all_items()
    print(f"Found {len(items)} items")

    # Stack imagery in mexican bounds at 10m resolution.
    stack = stackstac.stack(
        items,
        bounds=geomBounds(geom),
        snap_bounds=False,
        epsg=epsg,
        resolution=resolution,
        dtype="uint16",
        fill_value=0,
        resampling=Resampling.bilinear,
    )

    # Keep all bands with 10m or 20m resolution.
    stack = stack.sel(
        band=[
            "B04",
            "B03",
            "B02",
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

    filepath = "/home/tam/Desktop/videomap/videomap-{}-{}-{}-{}.mp4".format(
        *geomBounds(geom)
    )
    out = cv2.VideoWriter(
        filepath, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=4, frameSize=(size, size)
    )
    for img in composites:
        print("A")
        rgb = (
            numpy.clip(255 * img.data[:3] / 1500, 0, 255)
            .astype("uint8")
            .transpose(1, 2, 0)
        )
        out.write(rgb)
    out.release()

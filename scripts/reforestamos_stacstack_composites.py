import xarray

# import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.features import rasterize
import geopandas as gpd
import numpy

wd = Path("/home/tam/Documents/devseed")

epsg = 6362

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

BUFFER_SIZE_METERS = -20

total = len(list(wd.glob("geojson_sentinel/*.geojson")))

y, X = None, None
for counter, geojson in enumerate(wd.glob("geojson_sentinel/*.geojson")):
    print(f"Working on {counter + 1}/{total}")

    filepath = wd / "stacks" / f"{geojson.stem}.zarr"

    if not filepath.exists():
        continue

    data = xarray.open_zarr(filepath)

    scl = data.imagery.sel(band="SCL").astype("uint8")

    cloud_mask = scl.isin(CLOUDY_OR_NODATA)

    cleaned_data = data.imagery.where(~cloud_mask)

    # composites = cleaned_data.resample(time="14D", origin={"start": "2021-11-01", "end": "2022-04-30"}, closed="right").median("time")
    composites = cleaned_data.resample(
        time="14D", skipna=True, origin="2021-10-30", closed="right"
    ).median("time")

    # Rasterize the geometry with negative buffer
    src = gpd.read_file(geojson).to_crs(f"EPSG:{epsg}")
    rasterized = rasterize(
        [
            (dat.geometry.buffer(BUFFER_SIZE_METERS), CLASS_DN_LOOKUP[dat["class"]])
            for fid, dat in src.iterrows()
            if dat.geometry
        ],
        out_shape=composites.shape[2:],
        transform=composites.transform,
        all_touched=True,
        fill=0,
        dtype="uint8",
    )

    cdata = (
        composites.drop_sel({"band": "SCL"})
        .transpose("y", "x", "time", "band")
        .to_numpy()
    )
    cdata = cdata.reshape((-1, *cdata.shape[2:]))

    ydata = rasterized.ravel()

    cdata = cdata[ydata != 0]
    ydata = ydata[ydata != 0]

    if y is None:
        y = ydata
    else:
        y = numpy.hstack((y, ydata))

    if X is None:
        X = cdata
    else:
        X = numpy.vstack((X, cdata))
    break

    continue

    rgb = (
        (255 * composites.sel(band=["B04", "B03", "B02"]) / 3000)
        .clip(0, 255)
        .astype("uint8")
    )

    fig = plt.figure(figsize=(45, 50))
    imgx = 3
    imgy = 5
    for i in range(rgb.shape[0]):
        ax = fig.add_subplot(imgy, imgx, i + 1)
        xarray.plot.imshow(rgb[i], ax=ax)

    ax = fig.add_subplot(imgy, imgx, i + 2)
    data["training"] = (("y", "x"), rasterized)
    xarray.plot.imshow(data.training, ax=ax)

    plt.show()

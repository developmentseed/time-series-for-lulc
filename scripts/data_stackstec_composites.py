# import matplotlib.pyplot as plt
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray
from rasterio.features import rasterize
from wikidata import S2A_LULC_CLS

CLASS_DN_LOOKUP = {val: key for key, val in S2A_LULC_CLS.items()}

wd = Path("./data")
wd = Path("/home/tam/Desktop/aoi/tuxtla")

epsg = 6362

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

geojsons = list(wd.glob("geojson/*.geojson"))
total = len(geojsons)
geojsons.sort()

y, X = None, None
for counter, geojson in enumerate(geojsons):
    print(f"Working on {counter + 1}/{total}")

    filepath = wd / "stacks" / f"{geojson.stem}.zarr"

    if not filepath.exists():
        continue

    # if (wd / "cubes" / f"{geojson.stem}.npz").exists():
    #     continue

    data = xarray.open_zarr(filepath)

    scl = data.imagery.sel(band="SCL").astype("uint8")

    cloud_mask = scl.isin(CLOUDY_OR_NODATA)

    cleaned_data = data.imagery.where(~cloud_mask)

    # Create composites with cloud mask
    composites_using_cloud_mask = cleaned_data.resample(
        time="14D", skipna=True, origin="2021-10-30", closed="right"
    ).median("time")

    # Create composites without cloud mask
    composites_using_all_pixels = data.imagery.resample(
        time="14D", skipna=True, origin="2021-10-30", closed="right"
    ).median("time")

    # Fill pixels in cloud masked composites with pixels from full composite
    composites = xarray.where(
        np.isnan(composites_using_cloud_mask),
        composites_using_all_pixels,
        composites_using_cloud_mask,
    )

    del composites_using_all_pixels
    del composites_using_cloud_mask

    # # Rasterize the geometry with negative buffer
    # src = gpd.read_file(geojson).to_crs(f"EPSG:{epsg}")
    # rasterized = rasterize(
    #     [
    #         # (dat.geometry.buffer(BUFFER_SIZE_METERS), CLASS_DN_LOOKUP[dat["class"]])
    #         (dat.geometry, CLASS_DN_LOOKUP[dat["class"]])
    #         for fid, dat in src.iterrows()
    #         if dat.geometry
    #     ],
    #     out_shape=composites.shape[2:],
    #     transform=cleaned_data.transform,
    #     all_touched=True,
    #     fill=0,
    #     dtype="uint8",
    # )

    cdata = (
        composites.drop_sel({"band": "SCL"})
        .transpose("y", "x", "time", "band")
        .to_numpy()
    )
    del composites

    np.savez_compressed(wd / "cubesxy" / f"{geojson.stem}.npz", X=cdata.astype("uint16"), attrs=data.imagery.attrs)
    # np.savez_compressed(wd / "cubesxy" / f"{geojson.stem}.npz", X=cdata.astype("uint16"), attrs=data.imagery.attrs, y=rasterized)
    continue

    # cdata = cdata.reshape((-1, *cdata.shape[2:]))

    # ydata = rasterized.ravel()

    # cdata = cdata[ydata != 0]
    # ydata = ydata[ydata != 0]

    # if np.sum(np.isnan(cdata)):
    #     raise ValueError()

    # np.savez_compressed(wd / "cubesxy" / f"{geojson.stem}.npz", X=cdata.astype("uint16"), y=ydata.astype("uint8"))

    # continue

    # rgb = (
    #     (255 * composites.sel(band=["B04", "B03", "B02"]) / 3000)
    #     .clip(0, 255)
    #     .astype("uint8")
    # )

    # fig = plt.figure(figsize=(45, 50))
    # imgx = 3
    # imgy = 5
    # for i in range(rgb.shape[0]):
    #     ax = fig.add_subplot(imgy, imgx, i + 1)
    #     xarray.plot.imshow(rgb[i], ax=ax)

    # ax = fig.add_subplot(imgy, imgx, i + 2)
    # data["training"] = (("y", "x"), rasterized)
    # xarray.plot.imshow(data.training, ax=ax)

    # plt.show()

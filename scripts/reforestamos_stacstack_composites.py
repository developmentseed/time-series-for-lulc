import xarray
import matplotlib.pyplot as plt
from pathlib import Path

wd = Path("/home/tam/Documents/devseed")

geojson = list(wd.glob("geojson_sentinel/*.geojson"))[45]

filepath = wd / "stacks" / f"{geojson.stem}.zarr"

data = xarray.open_zarr(filepath)

scl = data.sel(band="SCL").astype("uint8")


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
cloud_mask = scl.isin(CLOUDY_OR_NODATA)

cleaned_data = data.where(cloud_mask)  # mask pixels where any one of those bits are set

# composites = cleaned_data.resample(time="14D", origin={"start": "2021-11-01", "end": "2022-04-30"}, closed="right").median("time")
composites = data.resample(
    time="14D", skipna=True, origin="2021-11-01", closed="right"
).median("time")
composites

rgb = (
    (255 * composites.imagery.sel(band=["B04", "B03", "B02"]) / 3000)
    .clip(0, 255)
    .astype("uint8")
)


fig = plt.figure(figsize=(45, 50))
imgx = 3
imgy = 5
for i in range(rgb.shape[0]):
    ax = fig.add_subplot(imgy, imgx, i + 1)
    xarray.plot.imshow(rgb[i], ax=ax)


plt.show()

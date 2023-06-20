from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray
from rasterio.features import rasterize
from wikidata import S2A_LULC_CLS
from joblib import Parallel, delayed
from tqdm import tqdm

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

def create_npz_files(geojson):
    # print(f"Working on {counter + 1}/{total}")

    zarr_filepath = wd / "stacks" / f"{geojson.stem}.zarr"

    if not zarr_filepath.exists():
        return

    # if (wd / "cubes" / f"{geojson.stem}.npz").exists():
    #     continue

    data = xarray.open_zarr(zarr_filepath)

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
    return 

# Run in parallel to reduce creation time
geojsons = list(wd.glob("geojson/*.geojson"))
geojsons.sort()
Parallel(n_jobs=-1)(
    delayed(create_npz_files)(geojson)
    for geojson in tqdm(geojsons, desc=f"Creating npz files", total=len(geojsons))
)

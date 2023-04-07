import concurrent.futures
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc
from dateutil.relativedelta import relativedelta
from shapely.geometry import box

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def compute_bbox_around_points(gdf, buffer=1000):
    gdf = gdf.to_crs(epsg=3857)
    gdf["bbox"] = gdf["geometry"].apply(lambda g: box(*g.buffer(1000).bounds))
    gdf = gdf.to_crs(epsg=4326)
    gdf.bbox = gdf.bbox.to_crs(epsg=4326)
    return gdf

def compute_date_range(date, delta):
    # Compute date range prior to delta
    prior_date = date + relativedelta(days=-delta)
    return f"{prior_date.strftime('%Y-%m-%d')}/{date.strftime('%Y-%m-%d')}" 


def download_image(sample):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=sample.bbox.bounds,
        datetime=compute_date_range(sample.date, 15),
        query=["eo:cloud_cover<25"]
    )
    
    print(len(search.get_all_items()))
    
    if len(search.get_all_items()):
        print("yes")
        items = list()
        #iterate over the results
        for item in search.get_all_items():
            item_bbox = box(*item.bbox)
            point_bbox = sample.bbox
            if item_bbox.contains(point_bbox):
                items.append(item)
        
        print(len(items))
        # sort items based on cloud_cover
        items = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])
        
        # save the image with least cloud_cover
        img = rioxarray.open_rasterio(items[0]\
                                    .assets["visual"].href)\
                                    .rio.clip_box(*sample.bbox.bounds, 
                                                  crs="EPSG:4326")
        print(img.data.shape)
        cv2.imwrite(f"{data_dir}/chips/visual-1km-radius/{sample['uid']}.png", img.data.transpose(1,2,0))


if __name__ == "__main__":
    cwd = Path(".")
    data_dir = Path("../data")
    gdf = gpd.read_file(data_dir/"label.geojson")
    gdf = compute_bbox_around_points(gdf)

    (data_dir/"chips"/"visual-1km-radius").mkdir(exist_ok=True, parents=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        samples = [s for _,s in list(gdf.iterrows())[:10]]
        executor.map(download_image, samples)
# LULC mapping using time series data & spectral bands

LULC mapping is difficult because of the complexity of the land cover types and the variability of the spectral signatures. This project aims to use time series data and spectral bands to map land cover types. 

In our recent work with Reforestamos, we faced a few challenges while trying to map different forest types in the corridors of Mexico; such as:

1. Data
- Generating training data for LULC mapping is expensive and time consuming
- Our data team spend ~150 hours in creatng just 1000 chips, all coming from the same mosaic & are annotated on clear days

2. Model
- Mosaics have very different data distribution and spectral signatures, it makes things difficult for the model to generalize
- Low clouds and shadows confuse the model for classes that have similar spectral signatures

In this project, we aim to solve these problems by using time series data and spectral bands. This approach has 2 advantages:
1. We only label areas where we are most confident about the class. We don't have to aasign a classes to every pixel, this reduces the time spent on labeling and increases the accuracy of the model.
2. We will use 1D convolutions that learn from time-series data. This will help the model generalize better and reduce the effect of clouds and shadows.

Because this is a simpler model, it takes less time to train and we can experiment faster.

As a part of labs POC:
- We will use already labelled data for Reforestamos using inward buffers to make sure the pixels are perfect representations of the class.
- We will use Sentinel Mosaics on a monthly basis to create the data cubes - time x spectral-bands x width x height
- Train 1D convolution model on the data cubes
- Evaluate the results & compare with the Unet/DeepLab model we have deployed in PEARL
- (Optional extended goal) - Use attention mechanism to improve the model

## Data

### Getting the training dataset

We have data for training reforestamos models hosted on Development Seed's S3 bucket.

```
aws s3 ls s3://ds-data-projects/reforestamos/reforestamos_sentinel/
                           PRE geojson/
                           PRE images_dt_12B/
                           PRE images_dt_4B/
                           PRE img_2020-03_2020-06/
                           PRE img_2020-06_2020-09/
                           PRE img_2020-09_2020-12/
                           PRE img_2020-12_2021-03/
                           PRE img_2021-03_2021-06/
                           PRE img_2021-06_2021-09/
                           PRE img_2021-09_2021-12/
                           PRE img_2022-06_2022-09/
                           PRE img_2022-12_2023-03/
                           PRE labels/
                           PRE no_rescale_color_formula/
2023-04-03 19:20:01      14340 .DS_Store
2023-04-03 19:20:01   57821721 clean_features.geojson
2023-03-04 16:10:56   42339856 clean_features.geojson.0bB292D3
2023-04-03 19:20:01      41808 clean_features.json
```

geojson/ - has vector masks
labels/ - has raster masks
images_dt_4B/ - has 4 band images
images_dt_12B/ - has 12 band images
no_rescale_color_formula/ - has 04 band images with no rescaling (by default, we rescale the images to 0,10000 while pulling the data from planetary computer)
*Ignore all the other folders*


### Code

I have added my nbs/ to the repository - this is where I was trying to pull data for HABs. Feel free to ignore this & start from scratch.





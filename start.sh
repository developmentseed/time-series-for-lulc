
# Download data created by data-team
aws s3 sync s3://ds-data-projects/reforestamos/reforestamos_sentinel/geojson/ data/geojson/

# Activate dev enviroment
conda activate time-series-for-lulc

# Read geojson files and create zarr files
# Result: zarr files in data/stacks/
python scripts/data_stackstac.py

# Read zarr files and create npz files
# Result: zarr files in data/cubesxy/
mkdir -p data/cubesxy/
python scripts/data_stackstec_composites.py

# Compare npz output
# Original files:s3://ds-labs-lulc/cubesxy/ downloaded in data/s3_cubesxy/
# Local Files generated: data/cubesxy
# aws s3 sync s3://ds-labs-lulc/cubesxy/ data/s3_cubesxy/
python scripts/compare_md5.py

aws s3 sync data/stacks/ s3://ds-labs-lulc/rub21/stacks/
aws s3 sync data/cubesxy/ s3://ds-labs-lulc/rub21/cubesxy/

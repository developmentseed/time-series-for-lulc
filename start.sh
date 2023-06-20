
# Download data created by data-team
aws s3 sync s3://ds-data-projects/reforestamos/reforestamos_sentinel/geojson/ data/geojson/

# Read geojson files and create zarr files
python scripts/reforestamos_stacstack.py

# Read zarr files and create npz files
mkdir -p data/cubexy/
python scripts/reforestamos_stacstack_composites.py

#Compare npz output files
aws s3 sync s3://ds-labs-lulc/cubesxy/ data/original_cubexy/
python scripts/compare_md5.py

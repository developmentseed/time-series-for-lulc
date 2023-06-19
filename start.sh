
# Download data created by data-team
aws s3 sync s3://ds-data-projects/reforestamos/reforestamos_sentinel/geojson/ data/geojson/

# Read geojson files and create zarr files
python scripts/reforestamos_stacstack.py

# Read zarr files and create npz files
pytohn scripts/reforestamos_stacstack_composites.py


# https://mlhub.earth/model/model-cv4a-crop-detection-v1
# https://lightning.ai/docs/pytorch/stable/deploy/production_advanced_2.html
# https://github.com/stac-extensions/ml-model/tree/main
# https://github.com/developmentseed/stacmlguide/discussions/2

torchserve --stop &&
torch-model-archiver --model-name tsmodel --version 1.0 --serialized-file /home/tam/Documents/repos/time-series-for-lulc/ts-model.pt --export-path /home/tam/Documents/repos/time-series-for-lulc/model-store --extra-files /home/tam/Documents/repos/time-series-for-lulc/inf/blight_handler.py --handler /home/tam/Documents/repos/time-series-for-lulc/inf/ts_handler.py -f &&
torchserve --start --model-store=/home/tam/Documents/repos/time-series-for-lulc/model-store --models tsmodel.mar --ts-config /home/tam/Documents/repos/time-series-for-lulc/inf/config.properties &&
python ./inf/post_tile.py

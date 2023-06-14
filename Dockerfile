FROM osgeo/gdal:latest
RUN apt update
RUN apt install -y python3-pip
# TODO add all dependencies in the requirements .txt file
RUN pip install rasterio 
RUN pip install httpx
RUN pip install Pillow
RUN pip install awscli
RUN pip install matplotlib
RUN pip install jupyter
RUN pip install jupyterlab
RUN pip install scikit-image
RUN pip install shapely
RUN pip install pyproj
RUN pip install xarray
RUN pip install geopandas
EXPOSE 8888
WORKDIR /home
CMD ["jupyter", "lab","--ip=0.0.0.0","--allow-root", "--notebook-dir=/home"]
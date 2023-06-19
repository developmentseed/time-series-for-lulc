FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml
RUN  conda update -n base -c defaults conda
RUN conda env create -f /tmp/environment.yml
ENV PATH /opt/conda/envs/time-series-for-lulc/bin:$PATH
WORKDIR /app
COPY . /app
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/app"]
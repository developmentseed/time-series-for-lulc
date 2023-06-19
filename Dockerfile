FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml

RUN  conda update -n base -c defaults conda

# Create the conda environment
RUN conda env create -f /tmp/environment.yml

# Activate the environment
ENV PATH /opt/conda/envs/my_docker_env/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy your project files into the container
COPY . /app
# CMD ["jupyter", "lab","--ip=0.0.0.0","--allow-root", "--notebook-dir=/home"]
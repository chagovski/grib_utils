# Use official Miniconda image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Create environment from environment.yml
RUN conda env create -f environment.yml

# Use the environment by default
ENV PATH /opt/conda/envs/ecmwf_forecasts/bin:$PATH

# Set entrypoint to run the script with arguments
ENTRYPOINT ["python", "src/ECMWF_forecasting_raster.py"]

# Example CMD, can be overridden at runtime
CMD ["/app/output"]

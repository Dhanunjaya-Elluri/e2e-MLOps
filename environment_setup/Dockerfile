# Build: docker build -t mlops_e2e_py310 .
FROM conda/miniconda3

# Copy environment file
COPY src/deployment/ci_dependencies.yml /setup/

# Activate environment
ENV PATH /opt/conda/envs/mlops_e2e_py310/bin:$PATH

# Install dependencies and activate environment
RUN conda update -n base -c defaults conda && \
    conda install python=3.10.11 && \
    conda create -f /setup/ci_dependencies.yml && \
    /bib/bash/ -c "source activate mlops_e2e_py310" && \
    az --version && \
    chmod -R 777 /usr/local/envs/mlops_e2e_py310/lib/python3.10/site-packages

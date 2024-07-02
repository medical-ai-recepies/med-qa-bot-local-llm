# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set non-interactive mode to avoid tzdata and other prompts
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install necessary packages
RUN apt-get update && \
    apt-get install -y git wget sudo bzip2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda based on the architecture
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
        /bin/bash /tmp/miniconda.sh -b -p /opt/conda; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && \
        /bin/bash /tmp/miniconda.sh -b -p /opt/conda; \
    fi && \
    rm /tmp/miniconda.sh

# Set up the path for conda
ENV PATH /opt/conda/bin:$PATH

# Install Python 3.12.1
RUN conda install -y python=3.12.1

# Re-install necessary packages for your app
RUN apt-get update && \
    apt-get install -y poppler-utils tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Accept the GitHub token as an argument
ARG GITHUB_TOKEN

# Clone the repository using the token
RUN git clone https://github.com/medical-ai-recepies/med-qa-bot-local-llm.git

# Change directory to the repository and checkout the dev branch
WORKDIR /med-qa-bot-local-llm
RUN git checkout dev
#Create a folder called logs
RUN mkdir logs

# Create and activate conda environment
RUN conda env create -f environment.yml --verbose
SHELL ["conda", "run", "-n", "medical_llm_env", "/bin/bash", "-c"]

# Expose the port
EXPOSE 8001

# Run the application
CMD conda run -n medical_llm_env python rag_google_scholar.py



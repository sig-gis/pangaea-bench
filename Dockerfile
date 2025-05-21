FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-3.py310
WORKDIR /


COPY environment.yaml .
COPY configs/ configs/
COPY pangaea/ pangaea/
COPY scripts/ scripts/

COPY setup.py .
COPY requirements.txt .


COPY README.md .
COPY tests/ tests/

RUN pip install -r requirements.txt
RUN pip install --no-build-isolation --no-deps -e .
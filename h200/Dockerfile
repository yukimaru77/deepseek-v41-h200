FROM lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155

# This image supplies the tested CUDA/PyTorch compiler toolchain. TensorFold
# runs from this repository; SGLang is not the serving engine.
WORKDIR /app
COPY h200/requirements.txt /app/h200/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/h200/requirements.txt
COPY src /app/src
COPY h200 /app/h200
COPY tools/dsv41 /app/tools/dsv41
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /app/
COPY LICENSES /app/LICENSES
COPY docs/H200-ATTRIBUTION.md /app/docs/H200-ATTRIBUTION.md
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 DATA_DIR=/data \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=8
ENTRYPOINT ["/bin/bash", "/app/h200/run.sh", "--inside-container"]

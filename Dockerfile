# syntax=docker/dockerfile:1
# Multi-stage build: wheels are built in a throwaway stage so the runtime image carries no
# compilers or build caches.
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 1000 evals
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
USER evals
WORKDIR /work
# Mount configs/datasets at /work and persist results with a volume on /work/.evalharness
ENTRYPOINT ["evalharness"]
CMD ["--help"]

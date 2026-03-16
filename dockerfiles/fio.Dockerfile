FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends fio && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["fio"]

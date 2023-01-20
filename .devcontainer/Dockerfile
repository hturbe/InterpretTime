ARG VARIANT="3.8.2"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

RUN apt update && apt install -y openjdk-11-jdk
RUN pip3 install pdm

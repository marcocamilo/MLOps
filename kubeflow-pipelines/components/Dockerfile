# Generated by KFP.

FROM python:3.10

WORKDIR /usr/local/src/kfp/components
COPY runtime-requirements.txt runtime-requirements.txt
RUN pip install --no-cache-dir -r runtime-requirements.txt

RUN pip install --no-cache-dir kfp==2.13.0
COPY . .

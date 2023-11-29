FROM python:3.11
ARG OPENAI_KEY
ENV OPENAI_KEY=$OPENAI_KEY
ENV PORT 8000

RUN apt-get update && apt-get install libgl1 -y

RUN curl -L https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite -o efficientdet.tflite


COPY requirements.txt /
RUN pip install -r requirements.txt

COPY ./src /src
#COPY .env /.env

CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}

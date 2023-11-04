# syntax=docker/dockerfile:1
 
FROM python:3.10.12-slim-bullseye
 
WORKDIR /work

EXPOSE 5000
ENV FLASK_APP=main.py
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt 
RUN pip3 install -r requirements.txt
COPY . .
 
ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0" ]

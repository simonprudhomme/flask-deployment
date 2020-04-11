# Pull Base Image
FROM ubuntu:16.04

# copy code into image and set as working directory
COPY . /app
WORKDIR /app

# install dependencies
RUN apt-get update -y
RUN apt-get install -y python3 python3-dev python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install flask gunicorn

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "server"]
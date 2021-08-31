FROM nvcr.io/nvidia/pytorch:20.06-py3

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY *.py ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

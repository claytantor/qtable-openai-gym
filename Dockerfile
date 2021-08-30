FROM tensorflow/tensorflow:1.8.0-gpu 

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

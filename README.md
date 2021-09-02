# using python 3
you need the ssl system packages because IOT requires ssl

```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

# building
docker build -t claytantor/qtable-openai-gym:latest .

# training

## with optimizization
```
docker run -d --gpus all --shm-size=1g --ulimit memlock=-1  --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace -e N_TRIALS=30 -e OPTUNA=True claytantor/qtable-openai-gym:latest python a2c_train.py
```

# predict
```
docker run --gpus all --shm-size=1g --ulimit memlock=-1  --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace claytantor/qtable-openai-gym:latest python a2c_predict.py
```


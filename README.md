# using python 3
you need the ssl system packages because IOT requires ssl

```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```


docker exec --gpus all --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace \
    claytantor/qtable-openai-gym:latest python anytrade.py 


docker exec claytantor/qtable-openai-gym:latest python anytrade.py
docker run --gpus all --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace \
    claytantor/qtable-openai-gym:latest python anytrade.py

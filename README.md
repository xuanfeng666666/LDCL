# LDCL

## Quick Start
### Environment variables & dependencies
```
PyTorch version: 2.2.1
CUDA version used by PyTorch: 12.1
DGL version: 2.2.1

conda create -n LDCL python=3.8
conda activate LDCL
```

### Process data
#### For the three datasets ICEWS18, ICEWS14, ICEWS05-15 and GDELT, go into the dataset folder in the ./data directory and run the following command
```
python get_his_subg.py

cd ./data/<dataset>
python ent2word.py
```

### Train models

#### Then the following commands can be used to train the proposed models. By default, dev set evaluation results will be printed when training terminates.

#### 1. Make dictionary to save models
```
mkdir models
```
#### 2. Train models
```
python src/main.py --dataset ICEWS14 --train-history-len 7 --test-history-len 7 --plm llama3b --model-type llama --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=1 --n-hidden 200 --self-loop --decoder seconvtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9  --pre-type all --add-static-graph  --temperature 0.03 --sem_temperature 0.03 --use-cl --alpha 0.7
```
#### 3. Test models
```
python src/main.py --dataset ICEWS14 --train-history-len 7 --test-history-len 7 --plm llama3b --model-type llama --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=1 --n-hidden 200 --self-loop --decoder seconvtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9  --pre-type all --add-static-graph  --temperature 0.03 --sem_temperature 0.03 --use-cl --alpha 0.7 --test
```

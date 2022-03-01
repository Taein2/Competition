python version : 3.8
CUDA:11.2

```
conda create -n comp2 python=3.8
```
```
conda activate comp2
```
```
pip install -r requirements.txt
```
```
cd code
```
``````
python demo.py 
--Transformation
TPS
--FeatureExtraction
ResNet
--SequenceModeling
BiLSTM
--Prediction
Attn
--image_folder
../datasets/test
--saved_model
saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth
--sensitive
```

datasets/test -> save result.csv & GTX.txt 

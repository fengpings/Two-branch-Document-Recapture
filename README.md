# Two-branch Document Recapture
 Implementation of ['Two-branch Multi-scale Deep Neural Network for Generalized Document Recapture'](https://arxiv.org/abs/2211.16786)
## Train
train model using ```python main.py```. For configuration of training settings, modify ```config/config.py```
## Test
test model using ```python main.py --test```
## Train/Val/Test data
structured like: 
```angular2html
root/train/recaptured/**.jpg
root/train/nonrecaptrued/**.jpg
...
```
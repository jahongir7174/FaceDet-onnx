Inference code of [SCRFD](https://arxiv.org/abs/2105.04714) using ONNX Runtime

### Installation

```
conda create -n ONNX python=3.8
conda activate ONNX
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install onnxruntime-gpu==1.14.0
pip install opencv-python==4.5.5.64
```

### Test

* Run `python main.py $1 $2` for testing, `$1` is model file path and `$2` is image file path

### Note

* This repo supports inference only, see reference for more details
* See `./weights/*` folder to check list of available models, `_kps` means the model includes 5 keypoints prediction

#### Reference

* https://github.com/deepinsight/insightface/tree/master/detection/scrfd

# Bakloska

Image analysis using combination of pre-trained neural networks.

## Instalation

1. Download pre-trained models [here](https://drive.google.com/file/d/1DYClIXxllr7h2veVj5Pmg9pXaPopIM-n/view?usp=sharing)
. Extract them into bakloska folder.
2. Create virtual environment 
```bash
python -m venv path_to_bakloska\venv
```
3. Install requirements.
```bash
pip install cmake
pip install -r requirements.txt
```

## Usage
```bash
# a single image with visualization
python main.py --image_path img/image.jpg --visualize True

# all images in dir without visualization
python main.py --image_path dir

# all images in img folder
python main.py
```  
To disable TensorFlow warnings add this at the beginning of predict.py file which should
be located at src/nsfw_detector/nsfw_detector/ or venv/src/nsfw_detector/nsfw_detector
```python
# Disabling Tensor Flow warnings
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```  


## Sources
[NSFW model](https://github.com/GantMan/nsfw_model)  
[Age_gender model](https://github.com/yu4u/age-gender-estimation)  
[Scene description model](https://github.com/CSAILVision/places365)  
[Object detection model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)  

## License
[MIT](https://choosealicense.com/licenses/mit/)

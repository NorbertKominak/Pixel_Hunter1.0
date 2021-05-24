# Bakloska

Image analysis using combination of freely available pre-trained neural networks with Google Cloud Vision, Amazon Rekognition and Microsoft Computer Vision APIs. 

Input images are loaded from the directory specified by --input_dir argument. All outputs are stored in the existing directory specified by --output_dir.Whether APIs are allowed to run or not is defined by --allow_api argument. Only selected images are sent for further analysis to APIs. The selection is determined by each neural network`s outputs.

## Installation

Python 3.7 or higher is required.

1. Download pre-trained models [here](https://drive.google.com/file/d/1DVcpRyNnuh-dS7y3gqniy0od9NZmkvy0/view?usp=sharing)
. Extract them into bakloska folder.
2. Create virtual environment, if you are stuck check [this](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

Windows:
```bash
# Set up the environment env in bakloska folder
python -m venv venv
# Activate created environment
venv\Scripts\activate

```

Linux and macOS:
```bash
# Set up the environment env in bakloska folder
python3 -m venv venv
# Activate created environment
source venv/bin/activate

```

3. Install requirements.
```bash
pip install cmake
pip install -r requirements.txt
```

In case dlib installation seems to be frozen, try to install it separately with verbose flags, to see whether it is really frozen. It may take a couple of minutes.
```bash
pip install dlib -vvv 
```

## Set up APIs Credentials
To get APIs running you need to give them your credentials. Manuals to obtain credentials:\
[Microsof Computer Vision](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/image-analysis-client-library?tabs=visual-studio&pivots=programming-language-python)\
[Google Cloud Vision](https://cloud.google.com/vision/docs/setup?authuser=0#windows)\
[Amazon Rekognition](https://docs.aws.amazon.com/rekognition/latest/dg/getting-started.html)

Put Microsoft\`s credentials at the top of (line 32) `msvision.py` file
```python
SUBSCRIPTION_KEY = "your_key"
ENDPOINT = "your_endpoint"
```

For Google\`s API, download the JSON file with yours service account key token and set os environment variable on the top of (line 29) `ggvision.py` file
```python
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path_to_the_json_token"
```

In case of the Amazon Rekognition, the easiest way is to download [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) and [configure](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html) it via command prompt. No changes with `awsvision.py` file are required.

You can still run the Image Analyzer(`run.py`) without APIs\` credentials with `--allow_api=False` (it is False by default). 

## Usage
```bash
# run analysis with APIs on images in my_dir and store all results
# to out_dir
python run.py --input_dir=my_dir --output_dir=out_dir --allow_api=True

# run analysis without APIs on images in img and store all results
# to outputs
python run.py
```  

[__Warning__]s with regards to custom gradient have no effect on object detection model\`s inference. It is only warning that custom gradients are not supported by TF and training this model would lead to an Error. More info [here](https://github.com/tensorflow/tensorflow/issues/44161)

## Sources
[NSFW model](https://github.com/GantMan/nsfw_model)  
[Age_gender model](https://github.com/yu4u/age-gender-estimation)  
[Scene description model](https://github.com/CSAILVision/places365)  
[Object detection model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)  

## License
[MIT](https://choosealicense.com/licenses/mit/)

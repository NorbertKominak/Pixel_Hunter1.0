from nsfw_detector import predict

MODEL_PATH = "models/nsfw.299x299.h5"
NETWORK_INPUT_SIZE = 299


def run(img_path):
    model = predict.load_model(MODEL_PATH)
    nsfw_classification = predict.classify(model, img_path, image_dim=NETWORK_INPUT_SIZE)
    return to_string(nsfw_classification)


def to_string(classification):
    with open("results/NSFW_detection.txt", "w") as output_file:
        for img_path, categories in classification.items():
            img_path = img_path[img_path.rfind('\\') + 1:]
            result = f'{img_path};\n'
            for category, value in categories.items():
                result += f'{category}: {(value * 100):.2f}%; '
            result += '\n'

            output_file.write(result)

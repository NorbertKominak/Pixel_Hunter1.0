from nsfw_detector import predict


def run(img_path):
    model = predict.load_model('nsfw.299x299.h5')
    nsfw_classification = predict.classify(model, img_path, image_dim=299)
    return to_string(nsfw_classification)


def to_string(classification):
    result = ''
    for img_path, categories in classification.items():
        img_path = img_path[img_path.rfind('\\') + 1:]
        result += f'{img_path};\n'
        for category, value in categories.items():
            result += f'{category}: {(value * 100):.2f}%; '
        result += '\n'
    return result

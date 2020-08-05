import os
import random

import cv2 as cv
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, json, render_template, request

from model import COVID19Net


model = COVID19Net()
model.load_state_dict(torch.load('./models/covid19_net.pth'))

app = Flask(__name__)


def clear_files(func):
    def wrapper():
        for image in os.listdir(os.path.join('static', 'images')):
            os.remove(os.path.join('static', 'images', image))
        res = func()
        return res
    wrapper.__name__ = func.__name__
    return wrapper


@app.route('/')
def index():
    if not os.path.exists(os.path.join('static', 'images')):
        os.makedirs(os.path.join('static', 'images'))
    return render_template('index.html')


@app.route('/api/diagnose', methods=['POST'])
@clear_files
def diagnose():
    image = request.files['image']
    if image and ('.' in image.filename and image.filename.rsplit('.', 1)[1] in {'jpeg', 'jpg', 'png'}):
        path = os.path.join('static', 'images', 'image') + str(random.randrange(1 << 16))
        image.save(path)
        image = cv.imread(path, 0)
        height, width = image.shape
        if height < width:
            left = ((width - height) // 2)
            image = image[:, left:left + height]
        else:
            top = ((height - width) // 2)
            image = image[top:top + width, :]
        image = cv.resize(image, (64, 64), interpolation=cv.INTER_CUBIC)
        image = Image.fromarray(image)
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            label = model(image)
        label = torch.round(label)
        label = int(label[0][0])
        return json.jsonify({
            'status': 'success',
            'label': label
        })
    else:
        return json.jsonify({
            'status': 'fail',
            'label': None
        })


if __name__ == '__main__':
    app.run()

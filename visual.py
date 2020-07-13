import os

import cv2 as cv
import numpy as np
from bokeh.plotting import figure, output_file, show
from sklearn.manifold import TSNE

if __name__ == '__main__':
    dataset = 'train'
    # dataset = 'test'
    X = []
    y = []

    for label in ['NORMAL', 'PNEUMONIA']:
        for filename in os.listdir(os.path.join('data', 'XRay', dataset, label)):
            try:
                image = cv.imread(os.path.join('data', 'XRay', dataset, label, filename), 0)
                X.append(image.flatten())
                y.append(0 if label == 'NORMAL' else 1)
            except:
                pass

    X = np.array(X)
    y = np.array(y)
    if dataset == 'train':
        idx = np.random.choice(5216, 624)
        X = X[idx]
        y = y[idx]

    X_embedded = TSNE(n_components=2).fit_transform(X)
    X_min, X_max = np.min(X_embedded), np.max(X_embedded)
    X_embedded = (X_embedded - X_min) / (X_max - X_min)

    colors = ['green', 'red']

    output_file('figure/train_data.html', title='Training Data')
    # output_file('figure/test_data.html', title='Testing Data')
    p = figure()
    p.scatter(X_embedded[:, 0], X_embedded[:, 1], radius=0.01, fill_color=[colors[c] for c in y], fill_alpha=0.5,
              line_color=None)
    show(p)

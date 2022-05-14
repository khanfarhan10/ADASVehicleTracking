# git clone  https://github.com/khanfarhan10/elm
# python setup.py install
# python EML/classification.py
import sklearn
import elm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import props

# SEEDS
np.random.seed(42)

print(f"Scikit Learn Version : f{sklearn.__version__}")

def run_EML_classification(X_image, y_labels):
    stdsc = StandardScaler()
    # **********************************
    # irises dataset classification
    # **********************************
    print("Image Properties")
    props(X_image)
    print("Label Properties")
    props(y_labels, show_uniques=True)
    print("RUNNING EML CLASSSIFICATION!")
    # load dataset
    # iris = load_iris()
    irx, iry = stdsc.fit_transform(X_image), y_labels
    print("irx shape:", irx.shape)
    print("iry shape:", iry.shape)
    x_train, x_test, y_train, y_test = train_test_split(irx, iry, test_size=0.2)

    # build model and train
    model = elm.elm(hidden_units=32, activation_function='relu',
                    random_type='normal', x=x_train, y=y_train, C=0.1, elm_type='clf')
    beta, train_accuracy, running_time = model.fit('solution_automated_vehicle_system')
    print("classifier beta:\n", beta)
    print("classifier train accuracy:", train_accuracy)
    print('classifier running time:', running_time)

    # test
    prediction = model.predict(x_test)
    print("classifier test prediction:", prediction)
    print('classifier test accuracy:', model.score(x_test, y_test))
    return prediction
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from scipy.ndimage import imread
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

img_src_path = "../cropped_images/"
wthr_filename = "../merged-data/merged-weather.csv"
wthr_filename_4_labels = "../merged-data-4-labels/merged-weather.csv"

image_dataframe = pd.DataFrame()


def read_weather_data():
    """ Read weather data from merged CSV file """
    # wthr_data = pd.read_csv(wthr_filename)
    wthr_data = pd.read_csv(wthr_filename_4_labels)
    return wthr_data  # .head(n=200)


def load_image(filename):
    """ Read image and reshape the images into a 1D array """
    pixels = imread(img_src_path + filename)
    # store the pixels as: r g b r g b r g b ...
    return pixels.reshape(pixels.shape[0] *
                          pixels.shape[1] *
                          pixels.shape[2])


def prepare_data():
    """ Return whole data: pixels in X and weather labels in y 
    """
    wthr_data = read_weather_data()
    wthr_data['pixels'] = wthr_data['Crspdng_Image'].apply(load_image)
    X = wthr_data['pixels'].values
    X = np.stack(X)
    y = wthr_data['Weather'].values[:, np.newaxis]
    return X, y


def make_first_model(X, y):
    """ PCA as transformer
        Multiclass classifier using LogisticRegression
    """
    y_binarized = binarize_y_values(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binarized)
    model = make_pipeline(
        PCA(350),
        OneVsRestClassifier(LogisticRegression(multi_class='ovr'))
    )
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


def make_second_model(X, y):
    """ PCA as transformer
        Using only first labels --> signleclass classification
        Using SVC
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y[:, 0])
    model = make_pipeline(
        StandardScaler(),
        PCA(350),
        SVC(kernel='linear', C=1)
    )
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


def make_third_model(X, y):
    """ PCA as transformer
        Multiclass classifier using SVC
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(
        StandardScaler(),
        PCA(500),
        SVC(kernel='linear', C=1e-2)
    )
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

def make_4th_model(X, y):
    """ PCA as transformer
        Multiclass classifier using SVC
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(
        PCA(500),
        KNeighborsClassifier(n_neighbors=7)
    )
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


def binarize_y_values(y):
    """ Use multilable binarization for y values """
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(y)


def main():
    """ Main function """
    X, y = prepare_data()
    # make_first_model(X, y)
    # make_second_model(X, y)
    # make_third_model(X, y)
    make_4th_model(X, y)


if __name__ == '__main__':
    main()

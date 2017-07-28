import seaborn
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


img_src_path = "../cropped_images/"
img_src_path2 = "../uncropped_images/"
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
    # pixels = imread(img_src_path2 + filename)
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
    """ PCA and StandardScaler as transformer
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
    """ PCA and StandardScaler as transformer
        classifier using SVC
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
        classifier using SVC
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(
        PCA(500),
        KNeighborsClassifier(n_neighbors=7)
    )
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


def make_5th_model(X, y):
    """ PCA as transformer
        Multiclass classifier using SVC
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(
        PCA(500),
        GaussianNB()
    )
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


def binarize_y_values(y):
    """ Use multilable binarization for y values """
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(y)


def report_label_predictions(X, y):
    """ PCA and StandardScaler as transformer
        classifier using SVC
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(
        StandardScaler(),
        PCA(500),
        SVC(kernel='linear', C=1e-2)
    )

    # count the number of 4 labels in y_test
    label_totals = count_labels(y_test)

    correct_predict_stats = {
        'Rain': 0,
        'Snow': 0,
        'Clear': 0,
        'Cloudy': 0,
    }

    # make predictions
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    # count the correct prediction for each label
    for i in range(y_predicted.shape[0]):
        if y_test[i][0] == y_predicted[i]:
            # count up correct prediction for the lable
            correct_predict_stats[y_predicted[i]] += 1

    # calculate the percentage of correct label predictions
    correct_predict_stats['Rain'] /= label_totals[0]
    correct_predict_stats['Snow'] /= label_totals[1]
    correct_predict_stats['Clear'] /= label_totals[2]
    correct_predict_stats['Cloudy'] /= label_totals[3]

    seaborn.set()
    plt.title('Weather Labels Classification Success Rate')
    plt.xlabel('Weather labels')
    plt.ylabel('Classification Success Rate (%)')
    x_labels = ['Clear', 'Cloudy', 'Rain', 'Snow']
    y_labels = [
        correct_predict_stats['Clear'] * 100,
        correct_predict_stats['Cloudy'] * 100,
        correct_predict_stats['Rain'] * 100,
        correct_predict_stats['Snow'] * 100
    ]

    seaborn.barplot(x=x_labels, y=y_labels, palette="Greens_d")
    plt.show()


def count_labels(labels):
    rain_count = 0
    snow_count = 0
    clear_count = 0
    cloudy_count = 0

    for label in labels:
            if label == 'Clear':
                clear_count += 1
            elif label == 'Cloudy':
                cloudy_count += 1
            elif label == 'Rain':
                rain_count += 1
            elif label == 'Snow':
                snow_count += 1
    return [rain_count, snow_count, clear_count, cloudy_count]


def main():
    """ Main function """
    X, y = prepare_data()
    # make_first_model(X, y)
    # make_second_model(X, y)
    # make_third_model(X, y)
    # make_4th_model(X, y)
    # make_5th_model(X, y)
    report_label_predictions(X, y)


if __name__ == '__main__':
    main()

import re
import pandas as pd
from os import listdir
from os.path import isfile, join

src_path = "../yvr-weather/"
dest_path = "../cleaned-weather-data/"
dest_path2 = "../cleaned-weather-data-4-labels/"

clear_re = re.compile(r'(.*)(Clear)(.*)')
cloudy_re = re.compile(r'(.*)(Cloudy|Fog)(.*)')
rain_re = re.compile(r'(.*)(Rain|Drizzle)(.*)')
snow_re = re.compile(r'(.*)(Snow|Ice)(.*)')

def read_ghcn_data(filename):
    """ Return the data from text and csv files """
    # only read Date/Time (as index) and Weather columns.
    raw_data = pd.read_csv(filename, index_col='Date/Time')
    return raw_data


def output_csv(data, filename):
    """ Write the data to a CSV file """
    # data.to_csv(dest_path + filename, header=True)
    data.to_csv(dest_path2 + filename, header=True)


def get_all_csv_filenames():
    image_files = [f for f in listdir(src_path) if isfile(join(src_path, f))]
    return image_files

def get_hour(time):
    s_time = str(time[:len(time)-3])
    return int(s_time)

def clean_data(data):
    """ Drop Datas:
            before 6 am and after 9pm
            drop data before 2016-06-05 
            drop data after 2017-06-21 
        Keep only(Date/Time) index and weather columns
        Drop Nan values for Weather
    """
    data['Time'] = data['Time'].apply(get_hour)
    # drop records with time before 6 am and after 9pm
    data = data[ (data['Time'] >= 6) & (data['Time'] <= 21)]
    only_weather = data['Weather']
    # print (only_weather.unique()) # debugging: see unique labels 
    only_weather = only_weather.apply(clean_wthr_labels)
    clean_data = only_weather.dropna()
    return clean_data

def clean_wthr_labels(label):
    """ Keep/Convert all labels to: 
        Clear, Cloudy, Rain, Snow
    """
    label = str(label).split(",")[0] # keep only first lable
    clear_match = clear_re.search(label)
    cloudy_match = cloudy_re.search(label)
    rain_match = rain_re.search(label)
    snow_match = snow_re.search(label)

    if clear_match:
        return 'Clear'
    elif cloudy_match:
        return 'Cloudy'
    elif rain_match:
        return 'Rain'
    elif snow_match:
        return 'Snow'
    else:
        return None

def main():
    """ Main function """
    csv_filenames = get_all_csv_filenames()

    for filename in csv_filenames:
        if filename[len(filename) - 3:] == 'csv':
            raw_data = read_ghcn_data(src_path + filename)
            final_data = clean_data(raw_data)
            output_csv(final_data, filename)

if __name__ == '__main__':
    main()

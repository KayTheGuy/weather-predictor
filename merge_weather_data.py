import pandas as pd
from os import listdir
from os.path import isfile, join

src_path = "../cleaned-weather-data/"
dest_path = "../merged-data/"
img_path  = "../cropped_images/"

def read_weather_data(filename):
    """ Return the weather data from csv file """
    raw_data = pd.read_csv(filename)
    return raw_data


def output_csv(data, filename):
    """ Write the data to a CSV file """
    data.to_csv(dest_path + filename, header=True)


def get_all_csv_filenames():
    """ Return all the csv filenames from source directory """
    image_files = [f for f in listdir(src_path) if isfile(join(src_path, f))]
    return image_files


def has_corresponding_image(filename):
    """ Check if the filename (date) has a corresponding image """
    if isfile(img_path + filename):
        return True
    else:
        return False


def match_image_filename(date):
    """ Read the date and format it to match the image file names """
    date = str(date)
    date = date.replace("-", "").replace(" ", "").replace(":", "")
    return date + '.jpg'


def main():
    """ Merge all weather data and keep only rows that has corresponding image """

    csv_filenames = get_all_csv_filenames()
    merged_data = pd.DataFrame()
    for filename in csv_filenames:
        if filename[len(filename) - 3:] == 'csv':
            csv_data = read_weather_data(src_path + filename)
            merged_data = merged_data.append(csv_data)

    merged_data['Crspdng_Image'] = merged_data['Date/Time'].apply(match_image_filename)
    merged_data['Has_Image'] = merged_data['Crspdng_Image'].apply(has_corresponding_image)
    merged_data = merged_data[merged_data['Has_Image'] == True]
    merged_data = merged_data.drop('Has_Image', 1)
    merged_data = merged_data.reset_index(drop=True)
    output_csv(merged_data, 'merged-weather.csv')

if __name__ == '__main__':
    main()

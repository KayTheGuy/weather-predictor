import pandas as pd
from os import listdir
from os.path import isfile, join

src_path = "../yvr-weather/"
dest_path = "../cleaned-weather-data/"


def read_ghcn_data(filename):
    """ Return the data from text and csv files """
    # only read Date/Time (as index) and Weather columns.
    ghcn_data = pd.read_csv(filename, usecols=[0, 24], index_col='Date/Time')
    return ghcn_data


def output_csv(data, filename):
    """ Write the data to a CSV file """
    data.to_csv(dest_path + filename, sep='\t')


def get_all_csv_filenames():
    image_files = [f for f in listdir(src_path) if isfile(join(src_path, f))]
    return image_files


def main():
    """ Main function """
    csv_filenames = get_all_csv_filenames()

    for filename in csv_filenames:
        if filename[len(filename) - 3:] == 'csv':
            ghcn_data = read_ghcn_data(src_path + filename)
            # drop NaN values
            ghcn_data = ghcn_data.dropna()
            output_csv(ghcn_data, filename)

if __name__ == '__main__':
    main()

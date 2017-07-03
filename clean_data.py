import sys
import numpy as np
import pandas as pd

path = "../yvr-weather/"

def read_ghcn_data(filename):
    """ Return the data from text and csv files """
    # read csv file
    ghcn_data = pd.read_csv(filename, header=None)
    return ghcn_data

def output_csv(data):
    """ Write the data to a CSV file """
    data.to_csv(out_csv, sep='\t')


def main():
    """ Main function """
    ghcn_data = read_ghcn_data(path + "201606.csv")
    print (ghcn_data)

if __name__ == '__main__':
    main()

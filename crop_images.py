import sys
from PIL import Image
from os import listdir
from os.path import isfile, join

src_path = "../katkam-scaled/"
dest_path = "../cropped_images/"


def crop_save_image(filename):
    """ crop the images to only capture the sky """
    img = Image.open(filename)
    area = (0, 0, 256, 129)
    cropped_img = img.crop(area)
    # to be used as index for next steps
    new_filename = filename[24:36] 
    cropped_img.save(dest_path + new_filename + ".jpg")


def get_all_image_filenames():
    image_files = [f for f in listdir(src_path) if isfile(join(src_path, f))]
    return image_files


def main():
    """ Main function """
    image_files = get_all_image_filenames()
    for img in image_files:
        crop_save_image(src_path + img)


if __name__ == '__main__':
    main()

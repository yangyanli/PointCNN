#!/usr/bin/python3
'''Download datasets for this project.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import gzip
import html
import shutil
import tarfile
import zipfile
import requests
import argparse
from tqdm import tqdm


# from https://gist.github.com/hrouault/1358474
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def download_from_url(url, dst):
    download = True
    if os.path.exists(dst):
        download = query_yes_no('Seems you have downloaded %s to %s, overwrite?' % (url, dst), default='no')
        if download:
            os.remove(dst)

    if download:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024 * 1024
        bars = total_size // chunk_size
        with open(dst, "wb") as handle:
            for data in tqdm(response.iter_content(chunk_size=chunk_size), total=bars, desc=url.split('/')[-1],
                             unit='M'):
                handle.write(data)


def download_and_unzip(url, root, dataset):
    folder = os.path.join(root, dataset)
    folder_zips = os.path.join(folder, 'zips')
    if not os.path.exists(folder_zips):
        os.makedirs(folder_zips)
    filename_zip = os.path.join(folder_zips, url.split('/')[-1])

    download_from_url(url, filename_zip)

    if filename_zip.endswith('.zip'):
        zip_ref = zipfile.ZipFile(filename_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
    elif filename_zip.endswith(('.tar.gz', '.tgz')):
        tarfile.open(name=filename_zip, mode="r:gz").extractall(folder)
    elif filename_zip.endswith('.gz'):
        filename_no_gz = filename_zip[:-3]
        with gzip.open(filename_zip, 'rb') as f_in, open(filename_no_gz, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder.')
    parser.add_argument('--dataset', '-d', help='Dataset to download.')
    args = parser.parse_args()
    print(args)

    root = args.folder if args.folder else '../../data'
    if args.dataset == 'tu_berlin':
        download_and_unzip('http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip', root,
                           args.dataset)
    elif args.dataset == 'modelnet':
        download_and_unzip('https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip', root, args.dataset)
        folder = os.path.join(root, args.dataset)
        folder_h5 = os.path.join(folder, 'modelnet40_ply_hdf5_2048')
        for filename in os.listdir(folder_h5):
            shutil.move(os.path.join(folder_h5, filename), os.path.join(folder, filename))
        shutil.rmtree(folder_h5)
    elif args.dataset == 'shapenet_partseg':
        download_and_unzip('https://shapenet.cs.stanford.edu/iccv17/partseg/train_data.zip', root, args.dataset)
        download_and_unzip('https://shapenet.cs.stanford.edu/iccv17/partseg/train_label.zip', root, args.dataset)
        download_and_unzip('https://shapenet.cs.stanford.edu/iccv17/partseg/val_data.zip', root, args.dataset)
        download_and_unzip('https://shapenet.cs.stanford.edu/iccv17/partseg/val_label.zip', root, args.dataset)
        download_and_unzip('https://shapenet.cs.stanford.edu/iccv17/partseg/test_data.zip', root, args.dataset)
        download_and_unzip('https://shapenet.cs.stanford.edu/iccv17/partseg/test_label.zip', root, args.dataset)
    elif args.dataset == 'mnist':
        download_and_unzip('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', root, args.dataset)
        download_and_unzip('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', root, args.dataset)
        download_and_unzip('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', root, args.dataset)
        download_and_unzip('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', root, args.dataset)
    elif args.dataset == 'cifar10':
        download_and_unzip('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', root, args.dataset)
    elif args.dataset == 'quick_draw':
        url_categories = 'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt'
        folder = os.path.join(root, args.dataset)
        folder_zips = os.path.join(folder, 'zips')
        if not os.path.exists(folder_zips):
            os.makedirs(folder_zips)
        filename_categories = os.path.join(folder_zips, url_categories.split('/')[-1])
        download_from_url(url_categories, filename_categories)

        categories = [line.strip() for line in open(filename_categories, 'r')]
        url_base = 'https://storage.googleapis.com/quickdraw_dataset/sketchrnn/'
        for category in categories:
            url = url_base + html.escape(category) + '.npz'
            filename_category = os.path.join(folder_zips, category + '.npz')
            download_from_url(url, filename_category)


if __name__ == '__main__':
    main()

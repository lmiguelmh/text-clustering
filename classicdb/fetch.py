"""
Loader and caching for the classic database,
Based on twenty_newsgroups.py

http://www.dataminingresearch.com/index.php/2010/09/classic3-classic4-datasets/
Classic3 and Classic4 DataSets
By Volkan TUNALI, September 10, 2010 4:53 am

The data is downloaded, extracted and cached in the '~/scikit_learn_data'
folder.

The `fetch_20newsgroups` function will not vectorize the data into numpy
arrays but the dataset lists the filenames of the posts and their categories
as target labels.

The `fetch_20newsgroups_vectorized` function will in addition do a simple
tf-idf vectorization step.

"""

import os
import logging
import tarfile
import pickle
import shutil
import re
import codecs

import numpy as np
import scipy.sparse as sp

from sklearn.datasets.base import get_data_home
from sklearn.datasets.base import Bunch
from sklearn.datasets.base import load_files
from sklearn.datasets.base import _pkl_filepath
from sklearn.utils import check_random_state
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.externals import joblib, six

if six.PY3:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen


logger = logging.getLogger(__name__)


# URL = ("http://www.dataminingresearch.com/download/dataset/classicdocs.rar")
URL = ("https://sites.google.com/site/xchgdir/public/classic.tar.gz?attredirects=0&d=1")
ARCHIVE_NAME = "classic.zip"
CACHE_NAME = "classic.pkz"
#ALL_FOLDER = "classic-all"
TRAIN_FOLDER = "classic-train"
TEST_FOLDER = "classic-test"
CLASSIC_HOME = "classic_home"

def download_classic(target_dir, cache_path):
    """Download the 20 newsgroups data and stored it as a zipped pickle."""
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)
    # all_path = os.path.join(target_dir, ALL_FOLDER)
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if os.path.exists(archive_path):
        # Download is not complete as the .tar.gz file is removed after
        # download.
        logger.warning("Download was incomplete, downloading again.")
        os.remove(archive_path)

    logger.warning("Downloading dataset from %s (1.5 MB)", URL)
    opener = urlopen(URL)
    with open(archive_path, 'wb') as f:
        f.write(opener.read())

    logger.info("Decompressing %s", archive_path)
    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)
    os.remove(archive_path)

    # Store a zipped pickle
    cache = dict(train=load_files(train_path, encoding='latin1'),
                 test=load_files(test_path, encoding='latin1'))
    # cache = dict(all=load_files(all_path, encoding='latin1'))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

    shutil.rmtree(target_dir)
    return cache


# def strip_newsgroup_header(text):
#     """
#     Given text in "news" format, strip the headers, by removing everything
#     before the first blank line.
#     """
#     _before, _blankline, after = text.partition('\n\n')
#     return after
#
#
# _QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
#                        r'|^In article|^Quoted from|^\||^>)')
#
#
# def strip_newsgroup_quoting(text):
#     """
#     Given text in "news" format, strip lines beginning with the quote
#     characters > or |, plus lines that often introduce a quoted section
#     (for example, because they contain the string 'writes:'.)
#     """
#     good_lines = [line for line in text.split('\n')
#                   if not _QUOTE_RE.search(line)]
#     return '\n'.join(good_lines)
#
#
# def strip_newsgroup_footer(text):
#     """
#     Given text in "news" format, attempt to remove a signature block.
#
#     As a rough heuristic, we assume that signatures are set apart by either
#     a blank line or a line made of hyphens, and that it is the last such line
#     in the file (disregarding blank lines at the end).
#     """
#     lines = text.strip().split('\n')
#     for line_num in range(len(lines) - 1, -1, -1):
#         line = lines[line_num]
#         if line.strip().strip('-') == '':
#             break
#
#     if line_num > 0:
#         return '\n'.join(lines[:line_num])
#     else:
#         return text


def fetch_classic(data_home=None, subset='all', categories=None,
                       shuffle=True, random_state=42,
                       remove=(),
                       download_if_missing=True):
    """Load the filenames and data from the 20 newsgroups dataset.

    Read more in the :ref:`User Guide <20newsgroups>`.

    Parameters
    ----------
    subset: 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    data_home: optional, default: None
        Specify a download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    categories: None or collection of string or unicode
        If None (default), load all the categories.
        If not None, list of category names to load (other categories
        ignored).

    shuffle: bool, optional
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state: numpy random number generator or seed integer
        Used to shuffle the dataset.

    download_if_missing: optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    remove: tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

        'headers' follows an exact standard; the other filters are not always
        correct.
    """

    data_home = get_data_home(data_home=data_home)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    classic_home = os.path.join(data_home, CLASSIC_HOME)
    cache = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(
                compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)

    if cache is None:
        if download_if_missing:
            cache = download_classic(target_dir=classic_home,
                                          cache_path=cache_path)
        else:
            raise IOError('classic dataset not found')

    if subset in ('train', 'test'):
        data = cache[subset]
    elif subset == 'all':
        data_lst = list()
        target = list()
        filenames = list()
        for subset in ('train', 'test'):
            data = cache[subset]
            data_lst.extend(data.data)
            target.extend(data.target)
            filenames.extend(data.filenames)

        data.data = data_lst
        data.target = np.array(target)
        data.filenames = np.array(filenames)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    data.description = 'the classic dataset'

    # if 'headers' in remove:
    #     data.data = [strip_newsgroup_header(text) for text in data.data]
    # if 'footers' in remove:
    #     data.data = [strip_newsgroup_footer(text) for text in data.data]
    # if 'quotes' in remove:
    #     data.data = [strip_newsgroup_quoting(text) for text in data.data]

    if categories is not None:
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        # Sort the categories to have the ordering of the labels
        labels.sort()
        labels, categories = zip(*labels)
        mask = np.in1d(data.target, labels)
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        # searchsorted to have continuous labels
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()

    return data


# def fetch_20newsgroups_vectorized(subset="train", remove=(), data_home=None):
#     """Load the 20 newsgroups dataset and transform it into tf-idf vectors.
#
#     This is a convenience function; the tf-idf transformation is done using the
#     default settings for `sklearn.feature_extraction.text.Vectorizer`. For more
#     advanced usage (stopword filtering, n-gram extraction, etc.), combine
#     fetch_20newsgroups with a custom `Vectorizer` or `CountVectorizer`.
#
#     Read more in the :ref:`User Guide <20newsgroups>`.
#
#     Parameters
#     ----------
#
#     subset: 'train' or 'test', 'all', optional
#         Select the dataset to load: 'train' for the training set, 'test'
#         for the test set, 'all' for both, with shuffled ordering.
#
#     data_home: optional, default: None
#         Specify an download and cache folder for the datasets. If None,
#         all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
#
#     remove: tuple
#         May contain any subset of ('headers', 'footers', 'quotes'). Each of
#         these are kinds of text that will be detected and removed from the
#         newsgroup posts, preventing classifiers from overfitting on
#         metadata.
#
#         'headers' removes newsgroup headers, 'footers' removes blocks at the
#         ends of posts that look like signatures, and 'quotes' removes lines
#         that appear to be quoting another post.
#
#     Returns
#     -------
#
#     bunch : Bunch object
#         bunch.data: sparse matrix, shape [n_samples, n_features]
#         bunch.target: array, shape [n_samples]
#         bunch.target_names: list, length [n_classes]
#     """
#     data_home = get_data_home(data_home=data_home)
#     filebase = '20newsgroup_vectorized'
#     if remove:
#         filebase += 'remove-' + ('-'.join(remove))
#     target_file = _pkl_filepath(data_home, filebase + ".pkl")
#
#     # we shuffle but use a fixed seed for the memoization
#     data_train = fetch_20newsgroups(data_home=data_home,
#                                     subset='train',
#                                     categories=None,
#                                     shuffle=True,
#                                     random_state=12,
#                                     remove=remove)
#
#     data_test = fetch_20newsgroups(data_home=data_home,
#                                    subset='test',
#                                    categories=None,
#                                    shuffle=True,
#                                    random_state=12,
#                                    remove=remove)
#
#     if os.path.exists(target_file):
#         X_train, X_test = joblib.load(target_file)
#     else:
#         vectorizer = CountVectorizer(dtype=np.int16)
#         X_train = vectorizer.fit_transform(data_train.data).tocsr()
#         X_test = vectorizer.transform(data_test.data).tocsr()
#         joblib.dump((X_train, X_test), target_file, compress=9)
#
#     # the data is stored as int16 for compactness
#     # but normalize needs floats
#     X_train = X_train.astype(np.float64)
#     X_test = X_test.astype(np.float64)
#     normalize(X_train, copy=False)
#     normalize(X_test, copy=False)
#
#     target_names = data_train.target_names
#
#     if subset == "train":
#         data = X_train
#         target = data_train.target
#     elif subset == "test":
#         data = X_test
#         target = data_test.target
#     elif subset == "all":
#         data = sp.vstack((X_train, X_test)).tocsr()
#         target = np.concatenate((data_train.target, data_test.target))
#     else:
#         raise ValueError("%r is not a valid subset: should be one of "
#                          "['train', 'test', 'all']" % subset)
#
#     return Bunch(data=data, target=target, target_names=target_names)

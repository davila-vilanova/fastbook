# from fastai.data.external import URLs, untar_data
# from fastai.metrics import accuracy
# from fastai.text.data import TextDataLoaders
# from fastai.text.learner import text_classifier_learner
# from fastai.text.models.core import AWD_LSTM

from fastbook import *
from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

learn.export()

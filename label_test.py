import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from PIL import Image

from utils.data import dataframe_from_tsv, dataframe_to_tsv, generator_from_dataframe, predict

def run():
    test_dataframe = dataframe_from_tsv(os.path.abspath('./data/test.tsv'))
    test_dataframe['label'] = ''

    train_dataframe = dataframe_from_tsv(os.path.abspath('./data/train.tsv'))
    train_generator = generator_from_dataframe(train_dataframe, batch_size=1, shuffle=False)

    classes_by_index=[''] * len(train_generator.class_indices)
    for label, idx in train_generator.class_indices.items():
        classes_by_index[idx] = label

    model = load_model('./picnic.model')
    for _, row in test_dataframe.iterrows():
        im = Image.open(row['file']).convert('RGB')
        class_ix = predict(im, model)
        row['label'] = classes_by_index[class_ix]

    dataframe_to_tsv(test_dataframe,'./data/labeled_test.tsv')


if __name__ == '__main__':
    run()
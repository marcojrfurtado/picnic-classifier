import pandas as pd
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array

def dataframe_from_tsv(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    if 'file' in df:
        dirname = os.path.dirname(tsv_file)
        basename = os.path.splitext(os.path.basename(tsv_file))[0]
        image_dir = os.path.join(dirname, basename)
        df['file'] = image_dir + '/' + df['file']
    return df


def dataframe_to_tsv(df, output_file):
    if 'file' in df:
        df['file'] = [os.path.basename(f) for f in df['file']]
    df.to_csv(output_file, sep='\t', index=False)


def generator_from_dataframe(df, batch_size=32, shuffle=True):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    return datagen.flow_from_dataframe(
        df,
        x_col='file',
        y_col='label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        color_mode='rgb'
    )

def predict(image, mnet_base_model):
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = mnet_base_model.predict(image)
    return np.argmax(yhat,axis=1)[0]

if __name__ == '__main__':
    dataframe = dataframe_from_tsv(os.path.abspath('../data/train.tsv'))
    print(dataframe)







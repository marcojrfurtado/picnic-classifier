from keras.applications import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from utils.data import dataframe_from_tsv, generator_from_dataframe
import os
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

OUTPUT_MODEL_FILENAME='picnic.model'

def append_layer(base_model,all_classes):
    nb_classes = len(np.unique(all_classes))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # dense layer 1
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    preds = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    return model

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

def run():
    train_dataframe = dataframe_from_tsv(os.path.abspath('./data/train.tsv'))
    train_generator = generator_from_dataframe(train_dataframe, shuffle=True)

    base_model = MobileNet(weights='imagenet', include_top=False)
    model = append_layer(base_model, train_dataframe['label'])

    setup_to_transfer_learn(model, base_model)

    step_size_train = train_generator.n // train_generator.batch_size
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=10)

    model.save(OUTPUT_MODEL_FILENAME)

if __name__ == '__main__':
    run()
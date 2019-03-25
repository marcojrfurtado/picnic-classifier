# Picnic Classifier

Employ transfer learning to retrain Resnet101 on the Picnic dataset.

## Requirements

* [Google Colaboratory](https://colab.research.google.com) account

## Upload data to Google Drive

Create a directory for your project on Google Drive. Move `train.ipynb` there.

Unpack the picnic dataset into the `data/` subdirectory, such that we have
```
<picnic drive directory>/data/train.tsv
<picnic drive directory>/data/test.tsv
<picnic drive directory>/data/test/
<picnic drive directory>/data/train/
```

## Enable GPU acceleration for notebook

Open `train.ipynb` on Google Colaboratory. 
Go to `Edit > Notebook Settings > Hardware Acceleration`, and select GPU.

## Training and labeling test set

Edit the first cell from `train.ipynb` to point out to the correct dataset directory. 
By default, it points to `picnic-classifier/data`. 
Run that cell to mount the Google Drive, you will need to type in an authorization code.

After mounting it, you can click to run all cells. After training is completed, script will
automatically label the test set using the trained model. Results will be written to `data/results.tsv`.

While training the model, Google Colaboratory will warn you that you are almost reaching the memory
limit of the GPU. You can ignore the warning, since the script will indeed use the GPU to the limit.

## Troubleshoot

### Receiving Input/Output error

If you receive the following message

```
OSError: [Errno 5] Input/output error: '/content/gdrive/My Drive/picnic-classifier/data/train/0.png'
```

, first double-check the directory provided in the first cell. 
If it is correct, simply try re-runing the notebook. 
Even after reporting a successful mount, Google Drive may still throw 
an error when trying access the data right away.
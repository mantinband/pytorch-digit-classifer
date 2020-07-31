# Digit classifier using pytorch framework

### Basic digit classifier using MNIST

First run will create, train and save the model, from there on the model will be loaded from storage.
To re-create the model simply delete the model file.

Expected output:
```
training model
100%|██████████| 6000/6000 [00:37<00:00, 159.66it/s]
epoch: 0, loss: 0.059727251529693604)
100%|██████████| 6000/6000 [00:36<00:00, 164.02it/s]
epoch: 1, loss: 0.004991937894374132)
100%|██████████| 6000/6000 [00:42<00:00, 141.32it/s]
epoch: 2, loss: 0.002053869189694524)
model trained! saving as 'digit-model.pt'..
saved!

testing test data set accuracy
100%|██████████| 1000/1000 [00:02<00:00, 394.79it/s]
model accuracy is: 96.39

testing train data set accuracy
100%|██████████| 6000/6000 [00:10<00:00, 547.65it/s]
model accuracy is: 97.55333333333334
```

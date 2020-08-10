# Biolab_SUM2020

### 8.5-8.12

- Test autoencoder with dim(bottleneck)= 2&3

- network structure:784-128-64-12-x-12-64-128-784

- hyperparameters:

  - epochs = 100

    batch_size = 128

    learning_rate = 1e-3

    loss = MSE

    optimizer=Adam

- - model.py: the autoencoder model, train&save
  - visualize.py: visualize the prediction of the output

  
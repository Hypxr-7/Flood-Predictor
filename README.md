# Flood Predictor

This is a project for our Introduction to AI course where we built a flood prediction model using ANNs and RNNs for anticipating floods in Pakistan and taking timely action

# Data Accusation

The prediction makes use of 4 attributes to make predictions

1. Land Surface Temperature (LST)
2. Normalized Difference Vegetation Index (NDVI)
3. Normalized Difference Snow Index (NDSI)
4. Precipitation
   
We used Google Earth Engine and its javascript API to acquire this data from 1-1-2000 to the present day.

For acquiring the labels, we used websites like wikipedia which contained the list of floods in Pakistan.

# Preprocessing the Data

The data as it was could not have been passed to our model for training. There were several things we needed to do to the data:

* Combine the various files for a province into a single one for each province
* Use an imputer to fill in missing values
* Combine each provincial file into a single one
* Add labels to the data

# Training and Validating

With the data in the right format, we could pass it into our model to train and test.

# Acknowledgments

This was a group project which could not have been done without the effort of the entire team:

* [Aadesh Panjwani](https://github.com/mobmuseum)
* [Syed Ahmed Farrukh](https://github.com/hydra4004)
* [Yaman Sibtain](https://github.com/Y-Sibtain)

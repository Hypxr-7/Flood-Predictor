# Flood Predictor

This is a project for our *Introduction to AI* course where we built a flood prediction model using Artificial Neural Networks (ANNs) and Recurrent Neural Networks (RNNs) to anticipate floods in Pakistan and enable timely action.

# Data Acquisition

Our model uses four key attributes to make predictions:

1. **Land Surface Temperature (LST)**
2. **Normalized Difference Vegetation Index (NDVI)**
3. **Normalized Difference Snow Index (NDSI)**
4. **Precipitation**

We used **Google Earth Engine** and its JavaScript API to acquire this data, covering the period from **January 1, 2000 to the present day**.

To obtain the flood occurrence labels, we consulted publicly available sources such as **Wikipedia**, which contains lists of major floods in Pakistan.

# Preprocessing the Data

The raw data required several preprocessing steps before it could be used to train our models:

* **Merging**: Combined multiple data files for each province into a single file per province.
* **Imputation**: Used imputers to fill in missing values.
* **Aggregation**: Merged all provincial files into one consolidated dataset.
* **Labeling**: Added flood occurrence labels to the data.

# Training and Validation

After preprocessing, we used the prepared dataset to train and validate both ANN and RNN models.

# Acknowledgments

This was a collaborative group project, made possible by the efforts of the entire team:

* [Aadesh Panjwani](https://github.com/mobmuseum)
* [Syed Ahmed Farrukh](https://github.com/hydra4004)
* [Yaman Sibtain](https://github.com/Y-Sibtain)

---

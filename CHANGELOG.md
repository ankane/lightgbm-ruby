## 0.4.3 (2025-06-26)

- Added ARM shared library for Linux

## 0.4.2 (2025-06-23)

- Improved performance of `predict` method for `pandas_categorical`

## 0.4.1 (2025-02-17)

- Updated LightGBM to 4.6.0

## 0.4.0 (2025-01-05)

- Added support for different prediction types
- Added support for `pandas_categorical` to `predict` method
- Added support for hashes and Rover data frames to `predict` method
- Added support for hashes to `Dataset`
- Added `importance_type` option to `dump_model`, `model_to_string`, and `save_model` methods
- Changed `Dataset` to use column names for feature names with Rover and Daru
- Changed `predict` method to match feature names with Daru
- Dropped support for Ruby < 3.1

## 0.3.4 (2024-07-28)

- Updated LightGBM to 4.5.0

## 0.3.3 (2024-06-15)

- Updated LightGBM to 4.4.0

## 0.3.2 (2024-01-25)

- Updated LightGBM to 4.3.0

## 0.3.1 (2023-09-13)

- Updated LightGBM to 4.1.0

## 0.3.0 (2023-07-22)

- Updated LightGBM to 4.0.0
- Fixed error with `dup` and `clone`
- Dropped support for Ruby < 3

## 0.2.7 (2023-02-01)

- Updated LightGBM to 3.3.5
- Improved ARM detection

## 0.2.6 (2021-10-24)

- Updated LightGBM to 3.3.0

## 0.2.5 (2021-07-07)

- Added `feature_name` method to boosters

## 0.2.4 (2021-03-26)

- Updated LightGBM to 3.2.0

## 0.2.3 (2021-03-09)

- Added ARM shared library for Mac

## 0.2.2 (2020-12-07)

- Updated LightGBM to 3.1.1

## 0.2.1 (2020-11-15)

- Updated LightGBM to 3.1.0

## 0.2.0 (2020-08-31)

- Updated LightGBM to 3.0.0
- Made `best_iteration` and `eval_hist` consistent with Python

## 0.1.9 (2020-06-10)

- Added support for Rover
- Improved performance of Numo datasets

## 0.1.8 (2020-05-09)

- Improved error message when OpenMP not found on Mac
- Fixed `Cannot add validation data` error

## 0.1.7 (2019-12-05)

- Updated LightGBM to 2.3.1
- Switched to doubles for datasets and predictions

## 0.1.6 (2019-09-29)

- Updated LightGBM to 2.3.0
- Fixed error with JRuby

## 0.1.5 (2019-09-03)

- Packaged LightGBM with gem
- Added support for missing values
- Added `feature_names` to datasets
- Fixed Daru training and prediction

## 0.1.4 (2019-08-19)

- Friendlier message when LightGBM not found
- Added `Ranker`
- Added early stopping to Scikit-Learn API
- Free memory when objects are destroyed
- Removed unreleased `dump_text` method

## 0.1.3 (2019-08-16)

- Added Scikit-Learn API
- Added support for Daru and Numo::NArray

## 0.1.2 (2019-08-15)

- Added `cv` method
- Added early stopping
- Fixed multiclass classification

## 0.1.1 (2019-08-14)

- Added training API
- Added many methods

## 0.1.0 (2019-08-13)

- First release

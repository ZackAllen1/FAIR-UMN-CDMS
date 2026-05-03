README: SuperCDMS Interaction Reconstruction via CNN
This directory contains the source code and data pipelines for implementing a Two-Branch Convolutional Neural Network designed to reconstruct the interaction positions of particles in SuperCDMS detectors. By utilizing high-precision timing markers, this model improves upon the baseline DNN-2 benchmark of 1.741 mm RMSE, achieving a Held-Out Subset (HOS) performance of 1.611 mm (a 7.4% improvement).

1. Directory Structure
   
  ```.
  ├── data/                       # Contains raw, split, and augmented temporal datasets.
  ├── plots/                      # Visualizations of training/validation loss curves and RMSE performance.
  ├── script_files/    
      ├── s1_augmentation.py      # Data augmentation
      ├── s2_split_and_save.py    # Splitting logic of the training, validation, HOS dataset
      ├── s3_cnn_dataprep.py      # Preparing the dataset for the CNN model
      ├── s4_twobranch_cnn.py     # The core architecture
      └── s5_train.py             # Training loop
   ```

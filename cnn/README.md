README: SuperCDMS Interaction Reconstruction via CNN
This directory contains the source code and data pipelines for implementing a Two-Branch Convolutional Neural Network designed to reconstruct the interaction positions of particles in SuperCDMS detectors. By utilizing high-precision timing markers, this model improves upon the baseline DNN-2 benchmark of 1.741 mm RMSE, achieving a Held-Out Subset (HOS) performance of 1.611 mm (a 7.4% improvement).

1. Directory Structure
   
  ```.
  .
  ├── data/                       # Contains raw, split, and augmented temporal datasets.
  ├── plots/                      # Visualizations of training/validation loss curves and RMSE performance.
  └── script_files/    
      ├── s1_augmentation.py      # Data augmentation
      ├── s2_split_and_save.py    # Splitting logic of the training, validation, HOS dataset
      ├── s3_cnn_dataprep.py      # Preparing the dataset for the CNN model
      ├── s4_twobranch_cnn.py     # The core architecture
      └── s5_train.py             # Training loop
```
2. Core Methodology
* **a. Feature Engineering:** Transformation of the reduced dataset into a \[N, 6, 16\] tensor comprising of Intra-Channel Deltas and Spatial Arrival Encoding
* **b. Two-Branch CNN:**
   * **i. Pulse Branch:** Employs a shared 1D-CNN encoder with weight sharing across all five detectors to learn universal pulse-shape statistics.
   * **ii. Spatial Branch:** An MLP that processes the 5-element arrival vector to determine the triangulation of the particle impact.
* **c. Mixup Training:** Uses a $\text{Beta}(\alpha, \alpha)$ sampling scheme to create synthetic intermediate positions, forcing the model to learn a continuous spatial mapping rather than memorizing the discrete training locations.

3. Performance & Results
  The model was evaluated on a held-out subset of positions (y = \[-12.502, -29.500, -41.900\] to ensure true generalization

| Metric | Value |
| ------ | ----- |
| Model Parameters | 4,769 |
| Benchmark RMSE | 1.741 mm |
|Our HOS RMSE | 1.611 mm |
|Improvement | 7.4% |

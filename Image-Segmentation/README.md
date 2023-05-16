# Image-Segmentation
This repository is a part of the work: *K-Means Clustering and its Application in Image Segmentation*, under the course: *EP4130/PH6130-Data Science Analysis*

## Introduction
This repository contains 2 python files: 
1. ImageSegmenter.py: File containing the class Image Segmentation with various attributes.
2. utils.py: File containing various utility functions supporting the Image Segmenter and for tools involving image processing.

## Requirements
This code is developed in Python 3.8.5 and requires the following packages:
1. numpy==1.23.5
2. matplotlib==3.7.1
3. scikit-learn==1.2.2

## ImageSegmenter.py
This class is used to segment an image into k clusters. 
Specify the number of clusters in the constructor.
### Methods:
- *fit:* This method takes an image as input and segments it into k clusters.
- *get_segmented_image:* This method returns the segmented image.
- *filter:* This method applies a custom filter to the input image that gives a vintage, noisy, painting-like apearance to the input image.
- *histogram:* This method generates the histogram for the image. If grayscale image, histogram for a single chanel is generated. If RGB image, histogram for each channel is generated.
- *normalize:* This method normalizes the image. If the image is grayscale, it normalizes the image. If the image is RGB, it normalizes each channel of the image.

### Class Variables:
- *num_clusters*: (int) Number of clusters to segment the image into.
- *kmeans*: (sklearn.cluster.KMeans) KMeans object from sklearn.cluster.
- *seg_image*: (numpy.ndarray) Segmented image.
- *cluster_centers*: (numpy.ndarray) Cluster centers of the segmented image.
- *colors*: (numpy.ndarray) Colors of the segmented image.

### Basic Usage:
For getting a segmentation of an image into 5 clusters:
```python
from ImageSegmenter import ImageSegmentation
import matplotlib.pyplot as plt

# Create an ImageSegmenter object
img_seg = ImageSegmentation(num_clusters=5)

# Load the image
img = plt.imread('image.jpg')

# Fit the image
img_seg.fit(img)

# Get the segmented image
seg_img = img_seg.get_segmented_image()

# Display the segmented image
plt.imshow(seg_img)
plt.show()
```

## utils.py
This file contains various utility functions for image processing.

### Functions:
1. *elbow_plot*: Generates the elbow curve for the given image to get the optimal number of clusters for K-Means.
```
Parameters
----------
image : (numpy.ndarray) The image to be segmented.
end : (int, optional) The number of clusters to be considered. The default is 10.
return_inertias : (bool, optional) Whether to return the inertias. The default is False.
return_times : (bool, optional) Whether to return the times taken. The default is False.

Returns
-------
Inertias and the times taken, if specified.
```

2. *silhouette_plot*: Generates the silhouette curve for the given image to get the optimal number of clusters for K-Means.
```
Parameters
----------
image : (numpy.ndarray) The image to be segmented.
end : (int, optional) The number of clusters to be considered. The default is 7.
return_scores : (bool, optional) Whether to return the scores. The default is False.
return_times : (bool, optional) Whether to return the times taken. The default is False.

Returns
-------
Scores and the times taken, if specified.
```

3. *optimal_k*:
    1. Finds the optimal value of K for the given image.
    2. Prints the optimal value of K and the respective graphs for the elbow and silhouette methods and the time graph vs the number of clusters.
```
Parameters
----------
image : (numpy.ndarray) The image to be segmented.
end : (int, optional) The number of clusters to be considered. The default is 10.
times : (bool, optional) Whether to return the times taken. The default is False.

Returns
-------
None.
```

4. *get_2norm_ratio*: Returns the ratio of the 2-norm of the closest point to the farthest point in a gaussian distribution of dimension dim.
```
Parameters
----------
dim : (int) Dimension of the gaussian distribution.

Returns
-------
(float) Ratio of the 2-norm of the closest point to the farthest point in a gaussian distribution of dimension dim.
```

5. *kl_divergence*: Returns the Kullback-Leibler divergence between two probability mass functions.
```
Parameters
----------
pmf1 : (numpy.ndarray) Probability mass function 1.
pmf2 : (numpy.ndarray) Probability mass function 2.

Returns
-------
(float) Kullback-Leibler divergence between pmf1 and pmf2.
```

6. awgn(img, sigma): Additive white gaussian noise
```
Parameters
----------
img : (numpy.ndarray) Input image
sigma : (float) Standard deviation of the noise

Returns
-------
(numpy.ndarray) Noisy image
```

7. add_outliers(image, n_outliers): Returns an image with n_outliers outliers. The image is normalized and then scaled to intensities between 0 and 255. The outlier intensity is calculated using the IQR of all the pixel intensities and then the outlier intensity is set as upper_limit + 5. We generate random coordinates and assign this outlier intensity to those pixels. The image is then normalized and scaled to intensities between 0-255. This simultes outliers in the pixel intensity space.
```
Parameters
----------
image : (numpy.ndarray) Image to add outliers to.

n_outliers : (int) Number of outliers to add to the image.

Returns
-------
(numpy.ndarray) Image with n_outliers outliers.
```

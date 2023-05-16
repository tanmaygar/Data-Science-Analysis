import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ImageSegmentation:

    ''' 
    This class is used to segment an image into k clusters. 
    Specify the number of clusters in the constructor.
    The class has two methods:
    1. fit: This method takes an image as input and segments it into k clusters.
    2. get_segmented_image: This method returns the segmented image.

    Class Variables
    ---------------
    num_clusters: (int) Number of clusters to segment the image into.
    kmeans: (sklearn.cluster.KMeans) KMeans object from sklearn.cluster.
    seg_image: (numpy.ndarray) Segmented image.
    cluster_centers: (numpy.ndarray) Cluster centers of the segmented image.
    colors: (numpy.ndarray) Colors of the segmented image.
    '''
    
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=10)
        self.seg_image = None
        self.cluster_centers = None
        self.colors = np.zeros((num_clusters, 1, 3), dtype=np.uint8)

    def normalize(self, image):
        
        '''
        This function normalizes the image.
        If the image is grayscale, it normalizes the image.
        If the image is RGB, it normalizes each channel of the image.

        Parameters
        ----------
        image: (numpy.ndarray) Image to be normalized.

        Returns
        -------
        Normalized image.
        '''
        if len(image.shape) < 3:
            return (image - np.min(image)) / (np.max(image) - np.min(image))
        
        elif len(image.shape) == 3:
            new = np.zeros(image.shape, dtype=np.float32)
            for k in range(3):
                new[:, :, k] = (image[:, :, k] - np.min(image[:, :, k])) / (np.max(image[:, :, k]) - np.min(image[:, :, k]))

            return new
    
    def fit(self, image):
        
        '''
        This function segments the image by clustering the image pixels into k clusters 
        based on the color intensity values of the pixel.

        Parameters
        ----------
        image: (numpy.ndarray) Image to be segmented.
        '''
        vectorized = np.float32(image.reshape((-1, 3)))
        self.kmeans.fit(vectorized)
        self.seg_image = self.kmeans.predict(vectorized).reshape(image.shape[:2])
        self.cluster_centers = np.array(self.kmeans.cluster_centers_, dtype=np.uint8)
        
        for i in range(self.num_clusters):
            self.colors[i, 0, 0] = self.cluster_centers[i, 0]
            self.colors[i, 0, 1] = self.cluster_centers[i, 1]
            self.colors[i, 0, 2] = self.cluster_centers[i, 2]

    def get_segmented_image(self):
        
        '''
        This function returns the segmented image with applying the colors of the 
        identified cluster centers.

        Returns
        -------
        Segmented image.
        '''
        new_image = np.zeros((self.seg_image.shape[0], self.seg_image.shape[1], 3), dtype=np.uint8)
        for i in range(self.num_clusters):
            new_image[self.seg_image == i, :] = self.colors[i]

        return new_image
    
    def filter(self, image, level = 6, return_img = False, show_img = True):
        
        '''
        Applies a custom filter to the input image that blurs the textures while
        maintaining the edges.
        This also displays the image.

        Parameters
        ----------
        image : (numpy.ndarray) The image to be filtered.
        level : (int, optional) The level of filtering. Lowest - 1, Highest - 10. The default is 6.
        return_img: (boolean, optional) States whether the generated image should be returned
                    or not. Default is False.
        show_img: (boolean, optional) States whether the generated image should be displayed
                    or not. Default is True.

        Returns
        -------
        (numpy.ndarray) The image with the filter applied on it.
        '''
        
        level = np.clip(level, 1, 10)    
        
        new = np.zeros(image.shape)
        noise = np.random.normal(0, 7, image.shape)
        new = image + noise
        new_image = np.array(self.normalize(new)*255, dtype = np.uint8)

        imgseg = ImageSegmentation(3 * level)
        imgseg.fit(new_image)
        segmented_image = imgseg.get_segmented_image()

        plt.imshow(segmented_image)
        # plt.imsave('processedImage.jpg', new)
        if show_img:
            plt.show()

        if return_img:
            return new

    def histogram(self, image, plot_hist=True, show_hist = True, return_hist = False):
        
        ''' 
        Generates the histogram for the image. If grayscale image, histogram for a 
        single chanel is generated. If RGB image, histogram for each channel is generated.

        Parameters
        ----------
        image: (numpy.ndarray) The image whose histogram is to be generated
        plot_hist: (boolean, optional) States whether the histograms should be plotted. 
                    Default is True. The plot is displayed only if show_hist is True.
        show_hist: (boolean, optional) States whether the histograms should be displayed.
        return_hist: (boolean, optional) States whether a list containing the histogram
                        for the constituent channels should be returned. Default is False.

        Returns
        -------
        out_hist: (list of numpy.ndarray) List containing the histograms of the constituent 
                    channels of the input image.
        '''
       
        out_hist = []
        if len(image.shape) == 2:
            
            if np.max(image) <= 1:
                fin_img = np.array(image*255, dtype = int)
            else:
                fin_img = image
            
            hist = np.zeros(256)
            for i in range(fin_img.shape[0]):
                for j in range(fin_img.shape[1]):
                    hist[int(fin_img[i, j])] += 1
            
            out_hist.append(hist)
        
        elif len(image.shape) == 3:
            
            r_hist = self.histogram(image[:, :, 0], False, False, True)[0]
            g_hist = self.histogram(image[:, :, 1], False, False, True)[0]
            b_hist = self.histogram(image[:, :, 2], False, False, True)[0]

            out_hist.append(r_hist)
            out_hist.append(g_hist)
            out_hist.append(b_hist)

        if plot_hist:
            if len(out_hist) == 2:
                plt.plot([i for i in range(0, 256)], out_hist[0])
            
            if len(out_hist) == 3:
                colors = ['r', 'g', 'b']
                for k in range(3):
                    plt.bar([i for i in range(256)], out_hist[k], width=1, color=colors[k], alpha=0.6, label=colors[k].upper()+' Channel')

            plt.legend()
            plt.grid()
            plt.title('Histogram')
            plt.xlabel('Intensities')
            plt.ylabel('Frequency')
        if show_hist:
            plt.show()
        if return_hist:
            return out_hist                             
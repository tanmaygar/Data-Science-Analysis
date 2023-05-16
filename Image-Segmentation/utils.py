import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from ImageSegmenter import ImageSegmentation

def elbow_plot(image, end=10, return_inertias=False, return_times=False):
    
    ''' 
    Generates the elbow curve for the given image.

    Parameters
    ----------
    image : (numpy.ndarray) The image to be segmented.
    end : (int, optional) The number of clusters to be considered. The default is 10.
    return_inertias : (bool, optional) Whether to return the inertias. The default is False.
    return_times : (bool, optional) Whether to return the times taken. The default is False.

    Returns
    -------
    Inertias and the times taken, if specified.
    '''
    
    inertias = []
    times = []
    
    for i in range(2, end+1):
        
        print('Elbow Method - Running for {} clusters...    '.format(i), end='\r')
        cur_inertia = 0
        cur_time = 0
        imgseg = ImageSegmentation(num_clusters = i)
        
        for j in range(3):
            st = time.time()
            imgseg.fit(image)
            cur_inertia += imgseg.kmeans.inertia_
            cur_time += time.time() - st
        
        inertias.append(cur_inertia / 3)
        times.append(cur_time / 3)
    print('Elbow Method - Completed.' + ' ' * 30)
    out = []
    if return_times:
        out.append(times)

    if return_inertias:
        out.append(inertias)

    if len(out) == 0:
        return
    else:
        return out


def silhouette_plot(image, end=7, return_scores=False, return_times=False):
    
    ''' 
    Generates the silhouette curve for the given image.

    Parameters
    ----------
    image : (numpy.ndarray) The image to be segmented.
    end : (int, optional) The number of clusters to be considered. The default is 7.
    return_scores : (bool, optional) Whether to return the scores. The default is False.
    return_times : (bool, optional) Whether to return the times taken. The default is False.

    Returns
    -------
    Scores and the times taken, if specified.
    '''
    
    X = image.reshape((-1, 3))
    range_n_clusters = [i for i in range(2, end+1)]
    scores = []
    times = []
    rows = int(np.ceil(len(range_n_clusters)/2))
    plot_num = 1
    plt.figure(figsize = (10, rows*4))
    for n_clusters in range_n_clusters:
        
        print('Silhouette Method - Running for {} clusters...    '.format(n_clusters), end='\r')
        ax = plt.subplot(rows, 2, plot_num)
        separator = len(X) // 50
        ax.set_xlim([-0.3, 1])
        ax.set_ylim([0, len(X) + (n_clusters + 1) * separator])
        local_scores = []
        local_times = []
        for i in range(3):
            st = time.time()
            clusterer = KMeans(n_clusters=n_clusters, n_init = 'auto')
            cluster_labels = clusterer.fit_predict(X)
            local_scores.append(silhouette_score(X, cluster_labels))
            local_times.append(time.time() - st)

        times.append(np.mean(local_times))
        silhouette_avg = np.mean(local_scores)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        # print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        scores.append(silhouette_avg)

        y_lower = separator
        for i in range(n_clusters):
            
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + separator

        ax.set_title("Silhouette plot for n_clusters = " + str(n_clusters) + ".")
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.grid()
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_yticks([])
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plot_num += 1
    
    print('Sihouette Method - Completed.' + ' ' * 30)
    plt.show()

    out = []
    if return_times:
        out.append(times)

    if return_scores:
        out.append(scores)

    if len(out) == 0:
        return
    else:
        return out


def optimal_k(image, end=10, times=False):
    
    ''' 
    1. Finds the optimal value of K for the given image.
    2. Prints the optimal value of K and the respective graphs for the elbow and silhouette methods
    and the time graph vs the number of clusters.

    Parameters
    ----------
    image : (numpy.ndarray) The image to be segmented.
    end : (int, optional) The number of clusters to be considered. The default is 10.
    times : (bool, optional) Whether to return the times taken. The default is False.

    Returns
    -------
    None.
    '''
    
    print('Running Elbow Method...', end='\r')
    elbow_out = elbow_plot(image, end, return_inertias=True, return_times=times)
    print('Running Silhouette Method...', end='\r')
    sil_out = silhouette_plot(image, end, return_scores=True, return_times=times)

    if times:
        elbow_times = elbow_out[0]
        sil_times = sil_out[0]
        inertias = elbow_out[1]
        scores = sil_out[1]

    else:
        inertias = elbow_out[0]
        scores = sil_out[0]
    
    print('ANALYSIS')
    print('--------\n')
    print('* Optimal Value of K is', np.argmax(scores)+2, 'with a score of {:5f}'.format(np.max(scores)), 'according to the Silhouette Method.')
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, end+1), inertias, 'ko-', label='Inertia')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertias')
    plt.title('Elbow Plot')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(range(2, end+1), scores, 'ko-', label='Silhouette score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.title('Silhouette Plot')
    plt.grid()
    plt.legend()
    plt.show()

    if times:
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        plt.plot(range(2, end+1), elbow_times, 'ko-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Time Taken')
        plt.title('Time for Elbow Method')
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(range(2, end+1), sil_times, 'ko-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Time Taken')
        plt.title('Time for Silhouette Method')
        plt.grid()
        plt.show()


def get_2norm_ratio(dim):
    
    ''' 
    Returns the ratio of the 2-norm of the closest point to the farthest point
    in a gaussian distribution of dimension dim.

    Parameters
    ----------
    dim : (int) Dimension of the gaussian distribution.

    Returns
    -------
    (float) Ratio of the 2-norm of the closest point to the farthest point
            in a gaussian distribution of dimension dim.
    '''
    
    mean = np.zeros(dim)
    cov = np.eye(dim)
    gaussian_sample = np.random.multivariate_normal(mean, cov, 1000)
    distances = []
    point = mean
    for point_ in gaussian_sample:
        distances.append(np.linalg.norm(point - point_))

    distances.sort()
    distances = distances[1:]

    return distances[0]/distances[-1]



def kl_divergence(pmf1, pmf2):
    '''' 
    Returns the Kullback-Leibler divergence between two probability mass functions.

    Parameters
    ----------
    pmf1 : (numpy.ndarray) Probability mass function 1.
    pmf2 : (numpy.ndarray) Probability mass function 2.
    
    Returns
    -------
    (float) Kullback-Leibler divergence between pmf1 and pmf2.
    '''
    
    div = 0
    for k in range(3):
        pmf1[:][k] /= np.sum(pmf1[:][k])
        pmf2[:][k] /= np.sum(pmf2[:][k])
        for j in range(len(pmf1[k])):
            if pmf1[k][j] != 0 and pmf2[k][j] != 0:
                div += pmf1[k][j]*np.log(pmf1[k][j]/pmf2[k][j])
    return div


def awgn(img, sigma):
    
    ''' 
    Additive white gaussian noise

    Parameters
    ----------
    img : (numpy.ndarray) Input image
    sigma : (float) Standard deviation of the noise
    
    Returns
    -------
    (numpy.ndarray) Noisy image
    '''
    
    imgseg = ImageSegmentation(2)
    new = np.zeros(img.shape)
    noise = np.random.normal(0, sigma, img.shape)
    new = img + noise
    return np.array(imgseg.normalize(new)*255, dtype = np.uint8)


def add_outliers(image, n_outliers):
    
    ''' 
    Returns an image with n_outliers outliers. The image is normalized and then 
    scaled to intensities between 0 and 255. The outlier intensity is calculated using
    the IQR of all the pixel intensities and then the outlier intensity is set as
    upper_limit + 5. We generate random coordinates and assign this outlier intensity
    to those pixels. The image is then normalized and scaled to intensities between 0-255.
    This simultes outliers in the pixel intensity space.

    Parameters
    ----------
    image : (numpy.ndarray) Image to add outliers to.

    n_outliers : (int) Number of outliers to add to the image.
    
    Returns
    -------
    (numpy.ndarray) Image with n_outliers outliers.
    '''
    
    new_image = image.copy()
    imgseg = ImageSegmentation(5)
    new_image = imgseg.normalize(new_image)
    new_image = np.array(new_image*230, dtype=np.uint8)
    rand_x = np.random.randint(0, image.shape[0], n_outliers)
    rand_y = np.random.randint(0, image.shape[1], n_outliers)
    rand_points = np.array([rand_x, rand_y]).T
    for point in rand_points:
        new_image[point[0]][point[1]] = np.array([255, 255, 255])
    return new_image
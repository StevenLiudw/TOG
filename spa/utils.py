from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt


def plot_img(img, title=None):
    plt.figure(figsize=(10,6))
    if title is not None:
        plt.title(title)
    plt.imshow(img)


def get_rotation_between_vecs(v1:np.ndarray, v2:np.ndarray)->np.ndarray:
    """
    Get the rotation between two vectors
    """
    v1 = np.expand_dims(v1, axis=0)
    v2 = np.expand_dims(v2, axis=0)
    mat, _ = R.align_vectors(v1, v2)
    return mat.as_matrix()
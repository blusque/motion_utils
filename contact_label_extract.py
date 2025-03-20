'''
Copyright (c) 2025, Jidong Mei
License: MIT (see LICENSE for details)

@author: Blusque (Jidong Mei)
@summary: This file provides utils to extract the contact labels from a given motion sequence.

The main algorithm is referenced from Strake et al. 2020.
The whole process can be splited into 5 steps:
1. calculate the 0/1 contact labels for a given endeffector.
2. normalize the 0/1 contact labels with the distribution of the motion fo the endeffector to form the continuous contact signal.
3. apply Butterworth filter to smooth the contact signal.
4. use triangular expansion to fit the contact signal frame by frame.
5. optimize the coefficients of the triangular expansion to get the final contact signal.
'''
from bvh import Animation
from scipy.signal import butter, filtfilt

class LBFGS(object):
    '''
    L-BFGS optimizer for optimizing the coefficients of the triangular expansion
    '''
    def __init__(self, max_iter=1000, gtol=1e-5, ftol=1e-5):
        self.max_iter = max_iter
        self.gtol = gtol
        self.ftol = ftol

def extract_contact_labels(animation, endeffector_ids, target_transforms, threshold=0.1):
    """
    Extract contact labels from a motion sequence.
    The main algorithm is referenced from Strake et al. 2020.
    The whole process can be splited into 5 steps:
    1. calculated

    Args:
        motion (np.ndarray): A motion sequence with shape (T, J, 3).
        endeffector_ids (list): A list of endeffector indices.
        target_transforms (list): A list of target transforms.
        threshold (float): The threshold for determining contact.

    Returns:
        np.ndarray: A contact label sequence with shape (T, J).
    """
    T, J, _ = motion.shape
    contact_labels = np.zeros((T, J), dtype=np.float32)
    for t in range(T):
        for j in endeffector_ids:
            if np.linalg.norm(motion[t, j] - target_transforms[j]) < threshold:
                contact_labels[t, j] = 1.0
    return contact_labels
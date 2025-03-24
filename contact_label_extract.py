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
import numpy as np

class LBFGS(object):
    '''
    L-BFGS optimizer for optimizing the coefficients of the triangular expansion
    '''
    def __init__(self, max_iter=1000, gtol=1e-5, ftol=1e-5):
        self.max_iter = max_iter
        self.gtol = gtol
        self.ftol = ftol

def extract_contact_labels(animation, endeffector_ids, target_translations, dist_threshold=0.05, vel_threshold=0.05):
    """
    Extract contact labels from a motion sequence.
    The main algorithm is referenced from Strake et al. 2020.
    The whole process can be splited into 5 steps:
    1. calculated

    Args:
        animation: Animation object
        endeffector_ids (list): A list of endeffector indices.
        target_translations (list): A list of target transforms.
        threshold (float): The threshold for determining contact.

    Returns:
        contact_labels: A contact label sequence with shape (T, len(ee)).
    """
    t, j, _ = animation.shape
    len_ee = len(endeffector_ids)
    if target_translations.ndim == 1:
        target_translations = np.expand_dims(target_translations, axis=(0, 1))
        target_translations.repeat(t, axis=0).repeat(len_ee, axis=1)
    
    translations = animation.translations
    ee_trans = translations[:, endeffector_ids]
    ee_vel = np.zeros_like(ee_trans)
    ee_vel[:-1] = np.diff(ee_trans, axis=0)

    assert ee_trans.shape == target_translations.shape, 'The shape of endeffector translations and target translations should be the same.'
    
    ee_trans_dist = np.linalg.norm(ee_trans - target_translations, axis=-1)
    ee_vel_norm = np.linalg.norm(ee_vel, axis=-1)
    contact_labels = (ee_trans_dist < dist_threshold).astype(np.float32) * (ee_vel_norm < vel_threshold).astype(np.float32)
    assert contact_labels.shape == (t, len_ee), 'The shape of contact labels is not correct.'
    return contact_labels.transpose(1, 0)
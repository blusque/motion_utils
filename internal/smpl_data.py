from tkinter import LEFT
from unittest.mock import Base
from attr import field
import numpy as np
from regex import P
import torch
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from smplx import SMPLX
from smplx.utils import SMPLOutput
from pydantic import BaseModel, Field
from dataclasses import dataclass

from .scene_data import ObjectData


class MotionTools:
    """
    Static utility class for motion data processing and feature extraction.
    Contains methods moved out from SMPLXMotionData to keep it focused on data representation.
    """
    
    @staticmethod
    def extract_foot_contact(left_foot_motions: np.ndarray, right_foot_motions: np.ndarray, threshold: float = 0.02) -> dict[str, np.ndarray]:
        """
        Extract foot contact information from foot motion data.
        
        Args:
            left_foot_motions (np.ndarray): Left foot positions, shape (num_frames, num_foot_joints, 3)
            right_foot_motions (np.ndarray): Right foot positions, shape (num_frames, num_foot_joints, 3) 
            threshold (float): Contact detection threshold
            
        Returns:
            dict[str, np.ndarray]: Dictionary containing foot contact information
        """
        num_frames = left_foot_motions.shape[0]
        
        # Calculate heights (Y-axis)
        left_foot_height = left_foot_motions[:, :, 1]   # (num_frames, num_foot_joints)
        right_foot_height = right_foot_motions[:, :, 1] # (num_frames, num_foot_joints)
        
        # Calculate velocities in XZ plane
        left_foot_xz = left_foot_motions[:, :, [0, 2]]  # (num_frames, num_foot_joints, 2)
        right_foot_xz = right_foot_motions[:, :, [0, 2]] # (num_frames, num_foot_joints, 2)
        
        left_foot_vel = np.concatenate([
            np.zeros((1, left_foot_xz.shape[1]), dtype=np.float32), 
            np.linalg.norm(np.diff(left_foot_xz, axis=0), axis=-1)
        ], axis=0)  # (num_frames, num_foot_joints)
        
        right_foot_vel = np.concatenate([
            np.zeros((1, right_foot_xz.shape[1]), dtype=np.float32), 
            np.linalg.norm(np.diff(right_foot_xz, axis=0), axis=-1)
        ], axis=0)  # (num_frames, num_foot_joints)
        
        # Determine contact based on height and velocity thresholds
        left_foot_contact = (left_foot_height < threshold).astype(np.int32) * (left_foot_vel < threshold).astype(np.int32)
        right_foot_contact = (right_foot_height < threshold).astype(np.int32) * (right_foot_vel < threshold).astype(np.int32)
        
        return {
            "left_foot_contact": left_foot_contact,
            "right_foot_contact": right_foot_contact,
            "combined_contact": np.concatenate([left_foot_contact, right_foot_contact], axis=1)
        }
    
    @staticmethod
    def extract_scene_object_contact(left_hand_motions: np.ndarray, right_hand_motions: np.ndarray, 
                                   hips_motions: np.ndarray, left_foot_motions: np.ndarray, 
                                   right_foot_motions: np.ndarray, object_data: ObjectData) -> dict[str, np.ndarray]:
        """
        Extract scene object contact information from motion data.
        
        Args:
            left_hand_motions (np.ndarray): Left hand positions, shape (num_frames, num_hand_joints, 3)
            right_hand_motions (np.ndarray): Right hand positions, shape (num_frames, num_hand_joints, 3)
            hips_motions (np.ndarray): Hip positions, shape (num_frames, num_hip_joints, 3)
            left_foot_motions (np.ndarray): Left foot positions, shape (num_frames, num_foot_joints, 3)
            right_foot_motions (np.ndarray): Right foot positions, shape (num_frames, num_foot_joints, 3)
            object_data (ObjectData): Object data containing position, rotation, and bounding box
            
        Returns:
            dict[str, np.ndarray]: Dictionary containing contact information
        """
        num_frames = left_hand_motions.shape[0]
        
        # Average positions for each body part
        left_hand_pos = np.mean(left_hand_motions, axis=1)  # (num_frames, 3)
        right_hand_pos = np.mean(right_hand_motions, axis=1)  # (num_frames, 3)
        
        # Transform to local object coordinates if rotation data is available
        if hasattr(object_data, 'rotations') and hasattr(object_data, 'positions'):
            if len(object_data.positions.shape) == 1:
                object_pos = np.tile(object_data.positions, (num_frames, 1))
            else:
                object_pos = object_data.positions
                
            local_left_hand = left_hand_pos - object_pos
            local_right_hand = right_hand_pos - object_pos
        else:
            local_left_hand = left_hand_pos
            local_right_hand = right_hand_pos
        
        # Check contact with object
        contact_label = np.zeros((num_frames, 2), dtype=np.int32)  # left, right hand
        contact_position = np.zeros((num_frames, 2, 3), dtype=np.float32)
        
        for i in range(num_frames):
            if hasattr(object_data, 'within_bounding_box'):
                left_contact = object_data.within_bounding_box(local_left_hand[i:i+1])
                right_contact = object_data.within_bounding_box(local_right_hand[i:i+1])
                contact_label[i, 0] = left_contact[0] if len(left_contact) > 0 else 0
                contact_label[i, 1] = right_contact[0] if len(right_contact) > 0 else 0
            
            contact_position[i, 0] = left_hand_pos[i]
            contact_position[i, 1] = right_hand_pos[i]
        
        return {
            "contact_label": contact_label,
            "contact_position": contact_position
        }
    
    @staticmethod
    def extract_trajectory_features(root_motions: np.ndarray, hips_motions: np.ndarray, 
                                  left_foot_motions: np.ndarray, right_foot_motions: np.ndarray,
                                  left_hand_motions: np.ndarray, right_hand_motions: np.ndarray) -> dict[str, np.ndarray]:
        """
        Extract trajectory features from motion data.
        
        Args:
            root_motions (np.ndarray): Root joint positions, shape (num_frames, 3)
            hips_motions (np.ndarray): Hip joint positions, shape (num_frames, num_hip_joints, 3)
            left_foot_motions (np.ndarray): Left foot positions, shape (num_frames, num_foot_joints, 3)
            right_foot_motions (np.ndarray): Right foot positions, shape (num_frames, num_foot_joints, 3)
            left_hand_motions (np.ndarray): Left hand positions, shape (num_frames, num_hand_joints, 3)
            right_hand_motions (np.ndarray): Right hand positions, shape (num_frames, num_hand_joints, 3)
            
        Returns:
            dict[str, np.ndarray]: Dictionary containing trajectory features
        """
        num_frames = root_motions.shape[0]
        
        # Project root trajectory to XZ plane
        trajectory_positions = root_motions[:, [0, 2]]  # (num_frames, 2)
        
        # Calculate velocities
        trajectory_velocities = np.concatenate([
            np.zeros((1, 2), dtype=np.float32), 
            np.diff(trajectory_positions, axis=0)
        ], axis=0)  # (num_frames, 2)
        
        # Calculate orientations from hip configuration
        if hips_motions.shape[1] >= 2:  # At least left and right hip
            left_hip = hips_motions[:, 0, :]   # Assume first is left hip
            right_hip = hips_motions[:, 1, :]  # Assume second is right hip
            hip_vector = right_hip - left_hip
            hip_vector_xz = hip_vector[:, [0, 2]]  # Project to XZ plane
            trajectory_orientations = np.arctan2(hip_vector_xz[:, 1], hip_vector_xz[:, 0])
        else:
            # Fallback: use velocity direction
            trajectory_orientations = np.arctan2(trajectory_velocities[:, 1], trajectory_velocities[:, 0])
        
        return {
            "trajectory_positions": trajectory_positions,
            "trajectory_velocities": trajectory_velocities,
            "trajectory_orientations": trajectory_orientations
        }


class SMPLXConfig(BaseModel):
    model_path: str = Field(default="./smplx", description="Path to the SMPLX model directory")
    gender: str = Field(default="neutral", description="Gender of the SMPLX model, options: 'neutral', 'male', 'female'")
    
    @classmethod
    def from_dataset_config(cls, dataset_config, gender: str = "neutral") -> 'SMPLXConfig':
        """Create SMPLXConfig from dataset configuration."""
        if hasattr(dataset_config, 'smplx_path'):
            model_path = dataset_config.smplx_path.get_model_path(gender)
            return cls(model_path=model_path, gender=gender)
        else:
            return cls(model_path="./smplx", gender=gender)


@dataclass
class MotionFeatures:
    """
    Represents the motion features extracted from the SMPLX motion data.
    """
    poses: np.ndarray   # shape (num_frames, num_joints * rot_dim + 3)
    trajectory: np.ndarray  # shape (num_frames, num_sample * 2)
    foot_contact: np.ndarray  # shape (num_frames, 4) for left and right foot contact
    scene_object_contact: np.ndarray  # shape (num_frames, num_objects, 3) for contact label


class SMPLXMotionData:
    """
    Represents the SMPLX motion data, including poses, betas, and translations.

    Attributes:
        poses (np.ndarray): The poses of the SMPLX model. Shape is (num_frames, num_joints * rot_dim).
        betas (np.ndarray): The shape parameters of the SMPLX model. Shape is (num_betas,).
        translations (np.ndarray): The translations of the SMPLX model. Shape is (num_frames, 3).
        frame_rate (int): The frame rate of the motion data.
        global_position (np.ndarray): The global positions of the SMPLX model. Shape is (num_frames, num_joints, 3).
        global_orientation (np.ndarray): The global orientations of the SMPLX model. Shape is (num_frames, num_joints, 4).
        rot_repr(str): The rotation representation used in the motion data, e.g., 'axis_angle', 'quaternion', 'rot6d'.

    Static Attributes:
        SKELETON_MAP (dict): A mapping of joint names to their indices in the pose array.
        SKELETON_NAMES (list): A list of joint names in the SMPLX model.
        UPPER_BONES (list): Indices of upper body bones.
        LOWER_BONES (list): Indices of lower body bones.
        HAND_BONES (list): Indices of hand bones.
        HEAD_BONES (list): Indices of head bones.
        LEFT_FOOT_BONES (list): Indices of left foot bones.
        RIGHT_FOOT_BONES (list): Indices of right foot bones.
        LEFT_HAND_BONES (list): Indices of left hand bones (hand only, no fingers).
        RIGHT_HAND_BONES (list): Indices of right hand bones (hand only, no fingers).
        LEFT_FINGER_BONES (list): Indices of left finger bones.
        RIGHT_FINGER_BONES (list): Indices of right finger bones.
        HIP_BONES (list): Indices of hip bones.
    
    Class Methods:
    - reset_first_frame_to_original_point: Initialize the first frame to the original point (0, 0, 0).
    - snap_to_ground: Snap the motion data to the ground by adjusting the translation.
    - convert_to_y_up_coordinate_system: Convert the motion data to a Y-up coordinate system. Y-up and -Z-forward, right hand coordinate system.
    
    Static Methods:
    - build_from_npz(file_path, smplx_path, filter: str = "savgol") -> 'SMPLXMotionData': Build the SMPLXMotionData from a npz file.
    - reconstruct_motion_data(parameters...) -> 'SMPLXMotionData': Reconstruct the SMPLXMotionData from a dictionary.
    """
    SKELETON_MAP = {
        "pelvis": 0,
        "left_hip": 1,
        "right_hip": 2,
        "spine1": 3,
        "left_knee": 4,
        "right_knee": 5,
        "spine2": 6,
        "left_ankle": 7,
        "right_ankle": 8,
        "spine3": 9,
        "left_foot": 10,
        "right_foot": 11,
        "neck": 12,
        "left_collar": 13,
        "right_collar": 14,
        "head": 15,
        "left_shoulder": 16,
        "right_shoulder": 17,
        "left_elbow": 18,
        "right_elbow": 19,
        "left_wrist": 20,
        "right_wrist": 21,
        "jaw": 22,
        "left_eye_smplhf": 23,
        "right_eye_smplhf": 24,
        "left_index1": 25,
        "left_index2": 26,
        "left_index3": 27,
        "left_middle1": 28,
        "left_middle2": 29,
        "left_middle3": 30,
        "left_pinky1": 31,
        "left_pinky2": 32,
        "left_pinky3": 33,
        "left_ring1": 34,
        "left_ring2": 35,
        "left_ring3": 36,
        "left_thumb1": 37,
        "left_thumb2": 38,
        "left_thumb3": 39,
        "right_index1": 40,
        "right_index2": 41,
        "right_index3": 42,
        "right_middle1": 43,
        "right_middle2": 44,
        "right_middle3": 45,
        "right_pinky1": 46,
        "right_pinky2": 47,
        "right_pinky3": 48,
        "right_ring1": 49,
        "right_ring2": 50,
        "right_ring3": 51,
        "right_thumb1": 52,
        "right_thumb2": 53,
        "right_thumb3": 54,
    }

    SKELETON_NAMES = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "jaw", "left_eye_smplhf", "right_eye_smplhf",
        "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3",
        "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3",
        "left_thumb1", "left_thumb2", "left_thumb3",
        "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3",
        "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3",
        "right_thumb1", "right_thumb2", "right_thumb3"
    ]

    PELVIS_BONE = [0]  # pelvis is the root joint

    BODY_BONES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # All body bones including hands and feet

    # Upper body bones (torso, arms, shoulders)
    UPPER_BONES = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # spine1, spine2, spine3, neck, collars, head, shoulders, elbows, wrists
    
    # Lower body bones (pelvis, legs, feet)
    LOWER_BONES = [0, 1, 2, 4, 5, 7, 8, 10, 11]  # pelvis, hips, knees, ankles, feet
    
    LEFT_HAND_BONES = [20]  # left wrist
    RIGHT_HAND_BONES = [21] # right wrist
    LEFT_FINGER_BONES = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]  # left hand fingers
    RIGHT_FINGER_BONES = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]  # right hand fingers
    LEFT_FOOT_BONES = [7, 10]  # left ankle, left foot
    RIGHT_FOOT_BONES = [8, 11]  # right ankle, right foot

    # Hand bones (all finger joints)
    HAND_BONES = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,  # left hand
                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]  # right hand
    
    # Head bones (jaw and eyes)
    HEAD_BONES = [22, 23, 24]  # jaw, left_eye, right_eye
    JAW_BONES = [22]  # jaw only
    LEFT_EYE_BONES = [23]  # left eye only
    RIGHT_EYE_BONES = [24]  # right eye only

    def __init__(self, data, cfg: SMPLXConfig):
        self.poses: np.ndarray = data['poses']
        self.poses = self.poses.reshape(-1, len(self.SKELETON_MAP), 3)        
        self.betas: np.ndarray = data['betas']
        self.trans: np.ndarray = data['trans']
        self.expressions: np.ndarray = data['expressions']
        self.frame_rate: int = data['frame_rate']
        self.frame_num: int = self.poses.shape[0] if self.poses is not None else 0
        self.model = SMPLX(
            model_path=cfg.model_path,
            gender=cfg.gender
        )
        self._recompute_global_info()
        self.init_first_frame_to_original_point()
        self.snap_to_ground()

    def init_first_frame_to_original_point(self):
        """
        Initialize the first frame to the original point (0, 0, 0).
        This is useful for aligning the motion data to a specific reference point.
        """
        # Get the pelvis position from the first frame
        init_position = self.global_position[0, self.SKELETON_MAP["pelvis"], :].numpy()
        self.trans -= init_position
        self._recompute_global_info()

    def snap_to_ground(self):
        """
        Snap the motion data to the ground by adjusting the translation.
        This is useful for ensuring that the feet are on the ground.
        """
        # Find the minimum height of the feet in the first frame
        left_foot_indices = self.LEFT_FOOT_BONES
        right_foot_indices = self.RIGHT_FOOT_BONES
        foot_indices = left_foot_indices + right_foot_indices
        min_height = torch.min(self.global_position[0, foot_indices, 1]).item()
        
        # Adjust the translation to snap to the ground
        self.trans[:, 1] -= min_height
        self._recompute_global_info()

    def convert_to_y_up_coordinate_system(self):
        """
        Convert the motion data to a Y-up coordinate system. Y-up and -Z-forward, right hand coordinate system.
        """
        # This is a placeholder - actual implementation would depend on the source coordinate system
        # For now, assume data is already in Y-up system
        pass

    def _recompute_global_info(self):
        """
        Recompute the global information after modifying the translation.
        This is necessary to update the global position and orientation.
        """
        # Extract body pose (first 21 joints, excluding global orientation)
        # SMPLX body_pose expects (batch_size, 21*3) = (batch_size, 63)
        body_pose = self.poses[:, self.BODY_BONES, :].reshape(self.frame_num, -1)  # Skip pelvis (joint 0), take joints 1-21
        left_hand_pose = self.poses[:, self.LEFT_FINGER_BONES, :].reshape(self.frame_num, -1)  # Left hand fingers
        right_hand_pose = self.poses[:, self.RIGHT_FINGER_BONES, :].reshape(self.frame_num, -1)  # Right hand fingers
        jaw_pose = self.poses[:, self.JAW_BONES, :].reshape(self.frame_num, -1)  # Jaw
        left_eye_pose = self.poses[:, self.LEFT_EYE_BONES, :].reshape(self.frame_num, -1)  # Left eye
        right_eye_pose = self.poses[:, self.RIGHT_EYE_BONES, :].reshape(self.frame_num, -1)  # Right eye
        
        # Extract global orientation (pelvis rotation)
        global_orient = self.poses[:, self.PELVIS_BONE, :].reshape(self.frame_num, -1)  # Pelvis rotation (joint 0)
        
        # Ensure proper tensor shapes - betas should be repeated for each frame
        betas_tensor = torch.from_numpy(self.betas).float()
        if betas_tensor.ndim == 1:
            betas_tensor = betas_tensor.unsqueeze(0).repeat(self.frame_num, 1)  # Shape: (frame_num, 10)
        
        self.global_info: SMPLOutput = self.model(
            betas=betas_tensor,
            global_orient=torch.from_numpy(global_orient).float(),
            body_pose=torch.from_numpy(body_pose).float(),
            left_hand_pose=torch.from_numpy(left_hand_pose).float(),
            right_hand_pose=torch.from_numpy(right_hand_pose).float(),
            transl=torch.from_numpy(self.trans).float(),
            expression=torch.from_numpy(self.expressions).float(),
            jaw_pose=torch.from_numpy(jaw_pose).float(),
            leye_pose=torch.from_numpy(left_eye_pose).float(),
            reye_pose=torch.from_numpy(right_eye_pose).float(),
        )
        self.global_position = self.global_info.joints
        self.global_orientation = self.global_info.global_orient

    @property
    def motion_data(self):
        return {
            "poses": self.poses,
            "betas": self.betas,
            "trans": self.trans,
            "frame_rate": self.frame_rate
        }
    
    @property
    def root_translation(self):
        return self.trans
    
    @property
    def upper_motion(self):
        return self.poses[:, self.UPPER_BONES, :]
    
    @property
    def lower_motion(self):
        return self.poses[:, self.LOWER_BONES, :]
    
    @property
    def hand_motion(self):
        return self.poses[:, self.HAND_BONES, :]
    
    @property
    def head_motion(self):
        return self.poses[:, self.HEAD_BONES, :]

    def get_root_rotations(self) -> np.ndarray:
        """
        Get the root rotations from the motion data.
        
        Returns:
            np.ndarray: The root rotations. shape (num_frames, 3)
        """
        # TODO: Implement the logic to extract root rotations by checking the hip rotation or the average rotation of the feet and shoulders.
        return self.poses[:, self.PELVIS_BONE, :]  # Assuming the first joint is the root joint
    
    """
    Please move all the extraction methods out of this class to the MotionTools class as static methods.
    This will help to keep the class focused on the motion data and its properties.
    After that, don't forget to reset their parameters, here's some possible parameters:
    extract_foot_contact(left_foot_motions: np.ndarray, right_foot_motions: np.ndarray) -> dict[str, np.ndarray]

    extract_scene_object_contact(left_hand_motions: np.ndarray, right_hand_motions: np.ndarray, hips_motions: np.ndarray, left_foot_motions: np.ndarray, right_foot_motions: np.ndarray, object_data: ObjectData) -> dict[str, np.ndarray]

    extract_trajectory_features(root_motions: np.ndarray, hips_motions: np.ndarray, left_foot_motions: np.ndarray, right_foot_motions: np.ndarray, left_hand_motions: np.ndarray, right_hand_motions: np.ndarray) -> dict[str, np.ndarray]
    """
    def extract_foot_contact(self):
        """
        Extract foot contact information from the motion data.
        The foot contact is determined by checking the height of the feet and their velocities. Please set the threshold to a reasonable value.
        The contact state is represented as 1 for contact and 0 for no contact. There are four joints for foot contact: left foot and right foot, each with two points (ankle and foot).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Left foot contact and right foot contact arrays.
            Shape is (num_frames, 2), (num_frames, 2).
        """
        left_foot_motions = self.global_position[:, self.LEFT_FOOT_BONES, :].numpy()
        right_foot_motions = self.global_position[:, self.RIGHT_FOOT_BONES, :].numpy()
        
        result = MotionTools.extract_foot_contact(left_foot_motions, right_foot_motions)
        return result["left_foot_contact"], result["right_foot_contact"]

    def extract_scene_object_contact(self, object_data: ObjectData):
        """
        Extract scene object contact information from the motion data.
        This is a placeholder implementation, as the actual contact detection logic depends on the scene and objects.
        Please do not implement this method temporarily, only change the input and output. The dataset is not ready yet.

        Returns:
            dict: A dictionary containing contact label, position, and relative position.
            The values are None as this is a placeholder implementation.
        """
        left_hand_motions = self.global_position[:, self.LEFT_HAND_BONES, :].numpy()
        right_hand_motions = self.global_position[:, self.RIGHT_HAND_BONES, :].numpy()
        hips_motions = self.global_position[:, [self.SKELETON_MAP["left_hip"], self.SKELETON_MAP["right_hip"]], :].numpy()
        left_foot_motions = self.global_position[:, self.LEFT_FOOT_BONES, :].numpy()
        right_foot_motions = self.global_position[:, self.RIGHT_FOOT_BONES, :].numpy()
        
        result = MotionTools.extract_scene_object_contact(
            left_hand_motions, right_hand_motions, hips_motions,
            left_foot_motions, right_foot_motions, object_data
        )
        return result["contact_label"], result["contact_position"]
    
    def extract_trajectory_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract trajectory features from the motion data.
        This function extracts the trajectory positions, velocities. The orientations should be extracted from average orientation of hips motion data.
        You should project the trajectory positions to the XZ plane. The orientations are represented as angles in radians of y-axis.
        
        Args:


        Returns:
            trajectory_positions (np.ndarray): The extracted trajectory positions. shape (num_frames, 2)
            trajectory_velocities (np.ndarray): The extracted trajectory velocities. shape (num_frames, 2)
            trajectory_orientations (np.ndarray): The extracted trajectory orientations. shape (num_frames,)
        """
        root_motions = self.global_position[:, self.SKELETON_MAP["pelvis"], :].numpy()
        hips_motions = self.global_position[:, [self.SKELETON_MAP["left_hip"], self.SKELETON_MAP["right_hip"]], :].numpy()
        left_foot_motions = self.global_position[:, self.LEFT_FOOT_BONES, :].numpy()
        right_foot_motions = self.global_position[:, self.RIGHT_FOOT_BONES, :].numpy()
        left_hand_motions = self.global_position[:, self.LEFT_HAND_BONES, :].numpy()
        right_hand_motions = self.global_position[:, self.RIGHT_HAND_BONES, :].numpy()
        
        trajectory = MotionTools.extract_trajectory_features(
            root_motions, hips_motions, left_foot_motions, 
            right_foot_motions, left_hand_motions, right_hand_motions
        )
        return trajectory['trajectory_positions'], trajectory['trajectory_velocities'], trajectory['trajectory_orientations']
        
    @staticmethod
    def build_from_npz(file_path, smplx_path=None, filter: str = "savgol", dataset_config=None, gender: str = "neutral") -> 'SMPLXMotionData':
        """
        Build the SMPLXMotionData from a npz file. You should move most functions in __init__ to this method. You should only leave the __init__ method to initialize the class variables.

        The function will load the motion data from the npz file, apply the specified filter for smoothing, and return an instance of SMPLXMotionData.

        1. Load the npz file containing the motion data.
        2. Apply the specified filter to smooth the motion data.
        3. Initialize the SMPLXMotionData instance with the loaded data.
        
        Args:
            file_path (str): The path to the npz file containing the motion data.
            smplx_path (str): The path to the SMPLX model directory (deprecated, use dataset_config instead).
            filter (str): The filter to use for smoothing the motion data. Options: 'savgol', 'gaussian'. Default is 'savgol'.
            dataset_config: Dataset configuration object containing SMPLX paths.
            gender (str): Gender for SMPLX model ('neutral', 'male', 'female').
        
        Returns:
            SMPLXMotionData: An instance of SMPLXMotionData with the loaded motion data.
        """
        # Load the npz file containing the motion data
        data = np.load(file_path)
        
        # Create data dictionary
        motion_data = {
            'poses': data['poses'],
            'betas': data['betas'] if 'betas' in data else np.zeros(10, dtype=np.float32),
            'trans': data['trans'] if 'trans' in data else np.zeros((data['poses'].shape[0], 3), dtype=np.float32),
            'expressions': data['expressions'] if 'expressions' in data else np.zeros((data['poses'].shape[0], 100), dtype=np.float32),
            'frame_rate': data['frame_rate'] if 'frame_rate' in data else 30
        }
        
        if filter not in ["savgol", "gaussian"]:
            raise ValueError(f"Unsupported filter: {filter}. Supported filters are 'savgol' and 'gaussian'.")
        # Apply the specified filter to the motion data
        if filter == "savgol":
            from scipy.signal import savgol_filter
            motion_data['poses'] = savgol_filter(motion_data['poses'], window_length=5, polyorder=2, axis=0)
            motion_data['trans'] = savgol_filter(motion_data['trans'], window_length=5, polyorder=2, axis=0)
        elif filter == "gaussian":
            from scipy.ndimage import gaussian_filter1d
            motion_data['poses'] = gaussian_filter1d(motion_data['poses'], sigma=1, axis=0)
            motion_data['trans'] = gaussian_filter1d(motion_data['trans'], sigma=1, axis=0)

        # Create SMPLX config from dataset config or fallback to smplx_path
        if dataset_config is not None:
            cfg = SMPLXConfig.from_dataset_config(dataset_config, gender)
        elif smplx_path is not None:
            cfg = SMPLXConfig(model_path=smplx_path, gender=gender)
        else:
            cfg = SMPLXConfig(model_path="./smplx", gender=gender)
        
        # Create and return SMPLXMotionData instance
        return SMPLXMotionData(motion_data, cfg, filter)


    @staticmethod
    def reconstruct_from_feature(upper_body_features: np.ndarray, lower_body_features: np.ndarray, 
                               hands_features: np.ndarray | None, expression_features: np.ndarray,
                               trajectory_velocities: np.ndarray, foot_contact: np.ndarray,
                               object_contact: np.ndarray, filter: str = "savgol", 
                               smplx_config: SMPLXConfig | None = None, 
                               dataset_config=None, gender: str = "neutral") -> 'SMPLXMotionData':
        """
        Understand the motion features you have extracted, and then reconstruct the motion data from the features.
        You should keep this function in the SMPLXMotionData class and as a static method to rebuild the features 
        Reconstruct the motion data from the features.
        Steps:
        1. reconstruct the motion data from the features.
            - Build the pose data from the body part features
            - Build the trajectory data by cumulating the velocities and orientations.
        2. use filters like savgol or gaussian to smooth the motion data.
        3. use the foot contact information to fix foot skating issues.
        4. (temporary not implement) use the object contact information to fix object interaction issues.

        Args:
            upper_body_features (np.ndarray): Pose feature of the upper body. shape (num_frames, num_upper_joints, 3)
            lower_body_features (np.ndarray): Pose feature of the lower body. shape (num_frames, num_lower_joints, 3)
            hands_features (np.ndarray | None): Pose feature of the hands, set to natural pose of smplx if not provided. shape (num_frames, num_hand_joints, 3)
            expression_features (np.ndarray): Expression feature, set to natural expressions if not provided. shape (num_frames, num_expression_joints, 103) (100 expressions + 3 jaw rotations)
            trajectory_velocities (np.ndarray): Trajectory velocities, the norm of the last axis is the speed and the direction is the global direction of the velocity. shape (num_frames, 2)
            foot_contact (np.ndarray): Foot contact information, shape (num_frames, 4) for left and right foot contact. Use this feature for foot skating fix.
            object_contact (np.ndarray): Object contact information, shape (num_frames, num_object_joints, 3). Use this feature for object interaction. Please do not implement this method temporarily, only change the input and output. The dataset and other surrounding implementations are not ready yet.
            filter (str): The filter to use for smoothing the motion data. Options: 'savgol', 'gaussian'. Default is 'savgol'.
            smplx_config (SMPLXConfig): Optional SMPLX configuration. If None, will use dataset_config or default.
            dataset_config: Optional dataset configuration object containing SMPLX paths.
            gender (str): Gender for SMPLX model ('neutral', 'male', 'female'). Default is 'neutral'.

        Returns:
            reconstructed_motion (SMPLXMotionData): The reconstructed motion data.
        """
        num_frames = upper_body_features.shape[0]
        
        # Step 1: Build the pose data from body part features
        # Initialize full pose array (55 joints * 3 for axis-angle representation)
        full_poses = np.zeros((num_frames, 55, 3), dtype=np.float32)
        
        # Assign upper body features to corresponding joints
        for i, joint_idx in enumerate(SMPLXMotionData.UPPER_BONES):
            if i < upper_body_features.shape[1]:
                full_poses[:, joint_idx, :] = upper_body_features[:, i, :]
        
        # Assign lower body features to corresponding joints
        for i, joint_idx in enumerate(SMPLXMotionData.LOWER_BONES):
            if i < lower_body_features.shape[1]:
                full_poses[:, joint_idx, :] = lower_body_features[:, i, :]
        
        # Assign hand features if provided, otherwise use natural pose (zeros)
        if hands_features is not None:
            for i, joint_idx in enumerate(SMPLXMotionData.HAND_BONES):
                if i < hands_features.shape[1]:
                    full_poses[:, joint_idx, :] = hands_features[:, i, :]
        
        # Step 2: Build trajectory data by cumulating velocities
        trajectory_positions = np.zeros((num_frames, 2), dtype=np.float32)
        for i in range(1, num_frames):
            trajectory_positions[i] = trajectory_positions[i-1] + trajectory_velocities[i]
        
        # Convert 2D trajectory to 3D translations (Y=0 for now, will be adjusted later)
        translations = np.zeros((num_frames, 3), dtype=np.float32)
        translations[:, [0, 2]] = trajectory_positions  # X and Z from trajectory
        
        # Step 3: Apply smoothing filter
        poses_reshaped = full_poses.reshape(num_frames, -1)  # (num_frames, 55*3)
        
        if filter == "savgol":
            window_length = min(5, num_frames if num_frames % 2 == 1 else num_frames - 1)
            if window_length >= 3:
                poses_reshaped = savgol_filter(poses_reshaped, window_length=window_length, polyorder=2, axis=0)
                translations = savgol_filter(translations, window_length=window_length, polyorder=2, axis=0)
        elif filter == "gaussian":
            sigma = 1.0
            poses_reshaped = gaussian_filter1d(poses_reshaped, sigma=sigma, axis=0)
            translations = gaussian_filter1d(translations, sigma=sigma, axis=0)
        
        # Step 4: Use foot contact information to fix foot skating
        # This is a simplified foot skating fix - lock feet to ground when in contact
        left_foot_contact = foot_contact[:, :2]  # First 2 columns for left foot
        right_foot_contact = foot_contact[:, 2:]  # Last 2 columns for right foot
        
        # For foot skating fix, we would need the global positions, but since we're reconstructing,
        # we'll apply a simple constraint by reducing motion when feet are in contact
        for frame in range(1, num_frames):
            # If left foot is in contact, reduce its motion
            if np.any(left_foot_contact[frame]):
                for joint_idx in SMPLXMotionData.LEFT_FOOT_BONES:
                    # Interpolate towards previous frame to reduce motion
                    full_poses[frame, joint_idx, :] = 0.7 * full_poses[frame, joint_idx, :] + 0.3 * full_poses[frame-1, joint_idx, :]
            
            # If right foot is in contact, reduce its motion
            if np.any(right_foot_contact[frame]):
                for joint_idx in SMPLXMotionData.RIGHT_FOOT_BONES:
                    # Interpolate towards previous frame to reduce motion
                    full_poses[frame, joint_idx, :] = 0.7 * full_poses[frame, joint_idx, :] + 0.3 * full_poses[frame-1, joint_idx, :]
        
        # Create motion data dictionary
        motion_data = {
            'poses': poses_reshaped,
            'betas': np.zeros(10, dtype=np.float32),  # Default shape parameters
            'trans': translations,
            'frame_rate': 30  # Default frame rate
        }
        
        # Use provided config or create from dataset config or fallback to default
        if smplx_config is not None:
            cfg = smplx_config
        elif dataset_config is not None:
            cfg = SMPLXConfig.from_dataset_config(dataset_config, gender)
        else:
            cfg = SMPLXConfig(model_path="./smplx", gender=gender)
        
        # Create and return SMPLXMotionData instance
        reconstructed_motion = SMPLXMotionData(motion_data, cfg, filter)
        
        return reconstructed_motion
import numpy as np
from scipy.spatial.transform import Rotation as R

from warnings import warn

order_map = {
    'Xrotation': 'X',
    'Yrotation': 'Y',
    'Zrotation': 'Z'
}

reverse_order_map = {
    'X': 'Xrotation',
    'Y': 'Yrotation',
    'Z': 'Zrotation'
}


def align_quat(qt: np.ndarray, inplace: bool):
    ''' make q_n and q_n+1 in the same semisphere
        the first axis of qt should be the time
    '''
    qt = np.asarray(qt)
    if qt.shape[-1] != 4:
        raise ValueError('qt has to be an array of quaterions')

    if not inplace:
        qt = qt.copy()

    if qt.size == 4:  # do nothing since there is only one quation
        return qt

    sign = np.sum(qt[:-1] * qt[1:], axis=-1)
    sign[sign < 0] = -1
    sign[sign >= 0] = 1
    sign = np.cumprod(sign, axis=0)

    qt[1:][sign < 0] *= -1
    return qt

class JointInfo:
    def __init__(self, name, offset, channels):
        self.name = name
        self.offset = offset
        self.channels = channels
        
class Joint:
    def __init__(self, info: JointInfo, idx):
        self.info = info
        self.idx = idx
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

class Skeleton:
    def __init__(self, root: Joint):
        self.root = root

    @staticmethod
    def build_skeleton(names, parents, offsets, channels):
        joints = []
        for i in range(len(names)):
            joint_info = JointInfo(names[i], offsets[i], channels[i])
            joint = Joint(joint_info, i)
            joints.append(joint)
        for i in range(len(parents)):
            if parents[i] == -1:
                root = joints[i]
            else:
                joints[parents[i]].children.append(joints[i])
        return Skeleton(root)
    
    def reset_root_by_name(self, root_name):
        new_root = self._reset_root_by_name(self.root, root_name)
        if new_root is None:
            raise ValueError('Root not found')
        self.root = new_root

    def _reset_root_by_name(self, joint: Joint, root_name):
        if joint.info.name == root_name:
            return joint
        
        for child in joint.children:
            new_root = self._reset_root_by_name(child, root_name)
            if new_root is not None:
                return new_root
        return None
    
    def pre_order_traversal(self, callback):
        self._pre_order_traversal(self.root, callback)

    def _pre_order_traversal(self, joint: Joint, callback):
        if joint.is_leaf():
            callback(joint)
            return
        
        callback(joint)
        for child in joint.children:
            self._pre_order_traversal(child, callback)
    
    @staticmethod
    def flatten_skeleton(joint: Joint):
        joints = []
        joints.append(joint)
        for child in joint.children:
            joints.extend(Skeleton.flatten_skeleton(child))
        return joints

class Animation:
    '''
    Animation data class.
    '''
    axis_map = {
        'x': 0,
        'y': 1,
        'z': 2
    }
    def __init__(self, root_idx, parents, offsets, channels, positions, rotations, frame_time=0.033333,
                 up_axis='y', forward_axis='z', order='XYZ'):
        '''
        Initialize the animation data.
        Input:
            root_idx: int
            parents: np.ndarray, shape=(num_joints,), dtype=int
            offsets: np.ndarray, shape=(1, num_joints, 3)
            positions: np.ndarray, shape=(num_frames, num_joints, 3)
            rotations: np.ndarray, shape=(num_frames, num_joints, 4)
            frame_time: float, optional, default=0.033333
        '''
        self.root_idx = root_idx
        self.parents = parents
        self.offsets = offsets
        self.channels = channels
        self.positions = positions
        self.rotations = rotations
        self.frame_time = frame_time
        self.num_frames = positions.shape[0]

        self.up_axis = up_axis
        self.forward_axis = forward_axis

        self.order = order
        self.reset()

    def auto_set_up_and_forward_axis(self):
        '''
        Automatically determine the up axis.

        Input:
            overwrite: bool, optional, default=False, whether to overwrite the existing axis.
        '''
        rest = self.rest()
        min_x = np.min(rest.translations[0, :, 0])
        min_y = np.min(rest.translations[0, :, 1])
        min_z = np.min(rest.translations[0, :, 2])
        max_x = np.max(rest.translations[0, :, 0])
        max_y = np.max(rest.translations[0, :, 1])
        max_z = np.max(rest.translations[0, :, 2])

        length_x = max_x - min_x
        length_y = max_y - min_y
        length_z = max_z - min_z

        max_axis = np.argmax([length_x, length_y, length_z])
        if max_axis == 0:
            self.up_axis = 'x'
        elif max_axis == 1:
            self.up_axis = 'y'
        elif max_axis == 2:
            self.up_axis = 'z'
        
        min_axis = np.argmin([length_x, length_y, length_z])
        if min_axis == 0:
            self.forward_axis = 'x'
        elif min_axis == 1:
            self.forward_axis = 'y'
        elif min_axis == 2:
            self.forward_axis = 'z'

    def reset(self):
        self.translations, self.orientations = self.fk()
        for i in range(len(self.parents)):
            self.rotations[:, i] = align_quat(self.rotations[:, i], True)
            self.orientations[:, i] = align_quat(self.orientations[:, i], True)

    def fk(self, positions=None, rotations=None):
        '''
        Forward kinematics.
        '''
        if positions is None:
            positions = self.positions.copy()
        if rotations is None:
            rotations = self.rotations.copy()

        translations = np.zeros_like(positions)
        orientations = np.zeros_like(rotations)
        for i in range(len(self.parents)):
            pi = self.parents[i]
            if pi == -1:
                translations[:, i] = positions[:, i]
                orientations[:, i] = rotations[:, i]
            else:
                offset = self.offsets[0, i]
                translations[:, i] = translations[:, pi] + R.from_quat(orientations[:, pi]).apply(offset)
                orientations[:, i] = (R.from_quat(orientations[:, pi]) * R.from_quat(rotations[:, i])).as_quat()

        return translations, orientations

    def __len__(self):
        return self.num_frames
    
    @property
    def num_joints(self):
        return len(self.parents)
    
    @property
    def data(self):
        return np.concatenate([self.positions, self.rotations], axis=2)
    
    def sub(self, start, end):
        return Animation(self.root_idx, self.parents, self.offsets, self.channels, self.positions[start:end], self.rotations[start:end], self.frame_time, self.up_axis, self.forward_axis, self.order)
    
    def _reparent(self, joint_list: list[int] | np.ndarray, root_idx=0):
        new_parents = np.zeros(len(joint_list), dtype=int)
        new_parents[root_idx] = -1
        for i in range(len(joint_list)):
            if i == root_idx:
                continue
            new_parents[i] = joint_list.index(self.parents[joint_list[i]])
        return new_parents
    
    def child(self, joint_list: list[int] | np.ndarray, root_idx=0):
        new_parents = self._reparent(joint_list, root_idx)
        new_channels = self.channels[joint_list].copy()
        new_channels[root_idx] = 6
        new_rotations = self.rotations[:, joint_list].copy()
        new_rotations[:, root_idx] = self.orientations[:, joint_list[root_idx]]
        return Animation(root_idx, new_parents, self.offsets[:1, joint_list].copy(), new_channels, self.positions[:, joint_list].copy(), new_rotations, self.frame_time, self.up_axis, self.forward_axis, self.order)
    
    def rest(self):
        rest_position = np.zeros_like(self.translations[:1])
        rest_rotation = np.zeros_like(self.rotations[:1])
        rest_rotation[..., -1] = 1
        rest_position[:, self.root_idx] = self.offsets[0, self.root_idx]
        rest_motion = Animation(self.root_idx, self.parents, self.offsets, self.channels, rest_position, rest_rotation, self.frame_time, self.up_axis, self.forward_axis, self.order)
        return rest_motion
    
    def offset_snap_to_ground(self):
        rest = self.rest()
        up_idx = Animation.axis_map[self.up_axis]
        rest_foot_position = np.min(rest.translations[0, :, up_idx])
        self.offsets[:, self.root_idx, up_idx] -= rest_foot_position
        self.reset()

    def motion_snap_to_ground(self):
        up_idx = Animation.axis_map[self.up_axis]
        lowest = np.min(self.translations[:, :, up_idx], axis=(0, 1))
        if abs(lowest) > 1e-6:
            print(f'Lowest: {lowest}')
        self.positions[:, self.root_idx, up_idx] -= lowest
        self.reset()
    
    def rotate_root(self, rotation: R):
        # self.offsets[0, 0] = rotation.apply(self.offsets[0, 0])
        self.positions[:, 0] = rotation.apply(self.positions[:, 0])
        self.rotations[:, 0] = (rotation * R.from_quat(self.rotations[:, 0])).as_quat()
        self.rotations[:, 0] = align_quat(self.rotations[:, 0], True)
        self.reset()

    def rotate_offset(self, up_axis: str):
        warn('This function is experimental and may not work as expected')
        warn("How to change the axis of the offset? We have to know that rotation doesn't work, if you don't believe it, try to")
        assert self.up_axis is not None, 'Up axis not set'
        assert up_axis in ['y', 'z'], 'Only support y and z'
        rotation = np.array([0, 0, 0])
        if up_axis == 'y':
            if self.up_axis == 'z':
                rotation[0] = -90
            elif self.up_axis == 'y':
                pass
            else:
                raise ValueError(f'{self.up_axis} to {up_axis} not implemented, only support z to y')
        elif up_axis == 'z':
            if self.up_axis == 'y':
                rotation[0] = 90
            elif self.up_axis == 'z':
                pass
            else:
                raise ValueError(f'{self.up_axis} to {up_axis} not implemented, only support y to z')
            
        rotation = R.from_euler('xyz', rotation, degrees=True)
        self.offsets[0] = rotation.apply(self.offsets[0])
        self.positions[:, 0] = rotation.apply(self.positions[:, 0])

        for i in range(len(self.parents)):
            self.rotations[:, i] = (rotation * R.from_quat(self.rotations[:, i]) * rotation.inv()).as_quat()
            # self.rotations[:, i] = align_quat(self.rotations[:, i], True)
        self.up_axis = up_axis
        self.reset()

class BVHParser:
    @staticmethod
    def find_bvh_files(folder):
        '''
        Find all BVH files in a folder.
        Input:
            folder: str

        Output:
            files: List[str]
        '''
        import os
        files = os.listdir(folder)
        bvh_files = []
        for file in files:
            if file.endswith('.bvh'):
                bvh_files.append(os.path.join(folder, file))
        return bvh_files
    
    @staticmethod
    def check_rotation_order(channels):
        '''
        Check the rotation order from the channels.
        Input:
            channels: List[str]

        Output:
            order: str
        '''
        order = ''
        for channel in channels:
            if 'rotation' in str.lower(channel):
                order += order_map[channel]
        return order

    @staticmethod
    def read_bvh(file_path, scale=1.0, order='XYZ', up_axis='y', forward_axis='z', given_offsets=None):
        '''
        Read a BVH file and return the root joint and motion data.
        Input:
            file_path: str
            order: str, optional, default='XYZ'
            up_axis: str, optional, default='y'
            forward_axis: str, optional, default='z'

        Output:
            names: List[str]
            animation_data: AnimationData
        '''
        with open(file_path, 'r') as f:
            lines = f.readlines()

        given_offsets = given_offsets or {}
        
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]

        # Read the hierarchy
        ptr = 0
        stack = []
        assert 'HIERARCHY' in lines[ptr], 'HIERARCHY not found'
        while 'MOTION' not in lines[ptr]:
            # Read the hierarchy
            if lines[ptr].split()[0] == 'HIERARCHY':
                names = []
                parents = []
                offsets = []
                channels = []
                ptr += 1
                continue

            # Read the root, three lines
            if lines[ptr].split()[0] == 'ROOT':
                root_idx = len(names)
                names.append('_'.join(lines[ptr].split()[1:])) # Root name
                parents.append(-1) # Root parent
                ptr += 1    # Next line
                if '{' in lines[ptr]:
                    stack.append(root_idx)
                    ptr += 1    # Skip '{'
                else:
                    continue
                if names[-1] in given_offsets.keys():
                    offsets.append(given_offsets[names[-1]])
                else:
                    offsets.append(np.array([float(word) for word in lines[ptr].split()[1:]]))  # Root offset
                ptr += 1    # Skip 'OFFSET'
                num_channels = int(lines[ptr].split()[1])
                assert num_channels in [0, 3, 6], 'Unknown number of channels %d' % num_channels
                assert len(lines[ptr].split()) == 2 + num_channels, 'Number of channels does not match'
                channels.append([word for word in lines[ptr].split()[2:]])  # Root channels
                new_order = BVHParser.check_rotation_order(channels[-1])
                if new_order != '' and order != new_order:
                    raise ValueError('Rotation order does not match')
                ptr += 1    # Skip 'CHANNELS'
            elif lines[ptr].split()[0] == 'JOINT':
                idx = len(names)
                names.append('_'.join(lines[ptr].split()[1:])) # Joint name
                parents.append(stack[-1])   # Parent is the last element in the stack
                ptr += 1    # Next line
                if '{' in lines[ptr]:
                    stack.append(idx)
                    ptr += 1    # Skip '{'
                else:
                    continue
                if names[-1] in given_offsets.keys():
                    offsets.append(given_offsets[names[-1]])
                else:
                    offsets.append(np.array([float(word) for word in lines[ptr].split()[1:]]))
                ptr += 1    # Skip 'OFFSET'
                num_channels = int(lines[ptr].split()[1])
                assert num_channels in [0, 3, 6], 'Unknown number of channels %d' % num_channels
                assert len(lines[ptr].split()) == 2 + num_channels, 'Number of channels does not match'
                channels.append([word for word in lines[ptr].split()[2:]])  # Joint channels
                new_order = BVHParser.check_rotation_order(channels[-1])
                if new_order != '' and order != new_order:
                    raise ValueError('Rotation order does not match')
                ptr += 1    # Skip 'CHANNELS'
            elif lines[ptr] == 'End Site':
                names.append('End_Site')
                parents.append(stack[-1])
                ptr += 1    # Skip 'End Site'
                ptr += 1    # Skip '{'
                offsets.append(np.array([float(word) for word in lines[ptr].split()[1:]]))
                channels.append([])
                ptr += 1    # Skip 'OFFSET'
                ptr += 1    # Skip '}'
            elif '}' in lines[ptr]:
                stack.pop()
                ptr += 1    # Skip '}'
            else:
                raise ValueError('Unknown line: %s' % lines[ptr])
        offsets = np.array(offsets).reshape(1, len(offsets), 3)
        parents = np.array(parents)
            
        # Read the motion data
        assert 'MOTION' in lines[ptr]
        positions = []
        rotations = []
        channel_nums = np.array([len(channel) for channel in channels])
        ptr += 1
        while ptr < len(lines):
            if 'Frames:' in lines[ptr]:
                num_frames = int(lines[ptr].split()[1])
            elif 'Frame Time:' in lines[ptr]:
                frame_time = float(lines[ptr].split()[2])
            else:
                record = [float(word) for word in lines[ptr].split()]
                current_positions = []
                current_rotations = []
                for channel in channels:
                    joint_rotation = np.zeros(3)
                    joint_position = np.zeros(3)
                    for i, channel_name in enumerate(channel):
                        if 'position' in str.lower(channel_name):
                            joint_position[i % 3] = record.pop(0)
                        elif 'rotation' in str.lower(channel_name):
                            joint_rotation[i % 3] = record.pop(0)
                    current_positions.append(joint_position)
                    current_rotations.append(joint_rotation)
                assert len(record) == 0, 'Record not empty'
                positions.append(np.array(current_positions))
                rotations.append(np.array(current_rotations))
            ptr += 1

        for i in range(len(rotations)):
            rotations[i] = R.from_euler(order, rotations[i], degrees=True).as_quat()
        positions = np.array(positions)
        rotations = np.array(rotations)

        offsets *= scale
        positions *= scale

        return names, Animation(root_idx, parents, offsets, channel_nums, positions, rotations, frame_time, up_axis, forward_axis, order)
    
    @staticmethod
    def slerp(t, q1, q2):
        '''
        Spherical linear interpolation.
        Input:
            t: float
            q1: np.ndarray, shape=(4,)
            q2: np.ndarray, shape=(4,)
        '''
        q1 = np.asarray(q1)
        q2 = np.asarray(q2)
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        dot = np.clip(dot, -1, 1)
        theta = np.arccos(dot) * t
        q = q2 - q1 * dot
        q = q / np.linalg.norm(q)
        return q1 * np.cos(theta) + q * np.sin(theta)


    @staticmethod
    def write_bvh(file_path, names, motion_data: Animation, order='XYZ'):
        '''
        Write a BVH file.
        Input:
            file_path: str
            names: List[str]
            channels: List[int]
            motion_data: AnimationData
            order: str, optional, default='XYZ'
        '''
        with open(file_path, 'w') as f:
            f.write('HIERARCHY\n')
            skeleton = Skeleton.build_skeleton(names, motion_data.parents, motion_data.offsets[0], motion_data.channels)
            BVHParser.write_joint(f, skeleton.root, 0, 'ROOT', order)
            f.write('MOTION\n')
            f.write('Frames: %d\n' % motion_data.num_frames)
            f.write('Frame Time: %f\n' % motion_data.frame_time)
            for frame in motion_data.data:
                frame_data = []
                for jid, joint in enumerate(frame):
                    if motion_data.channels[jid] == 6:
                        frame_data.extend(joint[:3])
                        frame_data.extend(R.from_quat(joint[3:]).as_euler(order, degrees=True))
                    elif motion_data.channels[jid] == 3:
                        frame_data.extend(R.from_quat(joint[3:]).as_euler(order, degrees=True))
                f.write(' '.join([str(value) for value in frame_data]) + '\n')

    @staticmethod
    def write_joint(f, joint: Joint, depth=0, type='ROOT', order='XYZ'):
        '''
        Write a joint to a BVH file.
        Input:
            f: file
            joint: Joint
            depth: int, optional, default=0
            type: str, optional, default='ROOT'
            order: str, optional, default='xyz'
        '''
        f.write('\t' * depth + f'{type} %s\n' % joint.info.name)
        f.write('\t' * depth + '{\n')
        f.write('\t' * (depth + 1) + 'OFFSET %f %f %f\n' % (joint.info.offset[0], joint.info.offset[1], joint.info.offset[2]))

        if (joint.info.channels == 6):
            channels = ' '.join(['Xposition', 'Yposition', 'Zposition'] + [reverse_order_map[order[i]] for i in range(3)])
        elif (joint.info.channels == 3):
            channels = ' '.join([reverse_order_map[order[i]] for i in range(3)])
        elif (joint.info.channels == 0):
            channels = ''
        else:
            raise ValueError('Unknown number of channels %d' % joint.info.channels)
        
        if joint.info.channels > 0:
            f.write('\t' * (depth + 1) + 'CHANNELS %d %s\n' % (joint.info.channels, channels))
        else:
            f.write('\t' * (depth + 1) + 'CHANNELS 0\n')
        
        if joint.is_leaf() and joint.info.name != 'End Site':
            f.write('\t' * (depth + 1) + 'End Site\n')
            f.write('\t' * (depth + 1) + '{\n')
            f.write('\t' * (depth + 2) + 'OFFSET %f %f %f\n' % (0, 0, 0))
            f.write('\t' * (depth + 1) + '}\n')
        for child in joint.children:
            if child.info.name == 'End Site':
                f.write('\t' * (depth + 1) + 'End Site\n')
                f.write('\t' * (depth + 1) + '{\n')
                f.write('\t' * (depth + 2) + 'OFFSET %f %f %f\n' % (child.info.offset[0], child.info.offset[1], child.info.offset[2]))
                f.write('\t' * (depth + 1) + '}\n')
            else:
                BVHParser.write_joint(f, child, depth + 1, 'JOINT')
        f.write('\t' * depth + '}\n')


if __name__ == '__main__':
    names, animation_data = BVHParser.read_bvh('test.bvh')
    BVHParser.write_bvh('output.bvh', names, animation_data)
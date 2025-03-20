import numpy as np
from kinematics import JacobianInverseKinematics
from bvh import Animation

def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r


def slerp(t, q1, q2, eps=1e-6):
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    theta = np.arccos(dot)
    if theta < eps:
        return q1
    elif theta < np.pi / 60.0:
        return (1 - t) * q1 + t * q2
    else:
        return (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)

def remove_foot_sliding(animation: Animation, foot, left_foot_id=(3, 4), right_foot_id=(7, 8), interp_length=10, force_on_floor=True):
    T = len(animation.translations)
    foot_id = list(left_foot_id) + list(right_foot_id)
    left_foot_id, right_foot_id = np.array(left_foot_id), np.array(right_foot_id)

    foot_heights = np.minimum(animation.translations[:, left_foot_id, 1],
                              animation.translations[:, right_foot_id, 1]).min(axis=1)  # [T, 2] -> [T]
    # print(np.min(foot_heights))
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    # print(floor_height)

    # default to y-up
    animation.translations[:, :, 1] -= floor_height    # snap the global transform to the floor
    animation.positions[:, 0, 1] -= floor_height    # displace the local root position

    for i, fidx in enumerate(foot_id):
        fixed = foot[i]  # [T]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = animation.translations[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += animation.translations[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                animation.translations[j, fidx] = avg.copy()

            # print(fixed[s - 1:t + 2])

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            animation.translations[s, fidx], animation.translations[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            animation.translations[s, fidx], animation.translations[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                animation.translations[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            animation.translations[s, fidx], animation.translations[l, fidx])
                animation.translations[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            animation.translations[s, fidx], animation.translations[r, fidx])
                animation.translations[s, fidx] = ritp.copy()

    targetmap = {}
    for j in range(animation.translations.shape[1]):
        targetmap[j] = animation.translations[:, j]
    # for fidx in fid:
    #     targetmap[fidx] = glb[:, fidx]

    ik = JacobianInverseKinematics(animation, targetmap, iterations=10, damping=5.0,
                                   silent=False)
    ik()

    return animation
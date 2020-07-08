from constants import *
from ransac import Ransac
from fundamentalmatrixmodel import FundamentalMatrixModel
import cv2
import copy
import json
import os
from scipy.optimize import least_squares


def calc_essential_matrix(K, fundamental_mat):
    essential_mat = K.T @ fundamental_mat @ K

    # 1)we are calculating essential_mat E this way assuming that the coordinates are normalized.
    # 2) also, rank(fundamental_mat)=2 => rank(essential_mat) should be 2.
    # based on the points (1 and 2) above, we will put(enforce) the singular values of E as 1,1,0

    U, _, VT = np.linalg.svd(essential_mat)
    S = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    essential_mat = U @ S @ VT
    #essential_mat /= np.linalg.norm(essential_mat)

    return essential_mat



def fundmntl_mat_from_8_point_ransac(point_list1, point_list2):
    fundamental_mat_model = FundamentalMatrixModel()
    ransac_model = Ransac(fundamental_mat_model)

    total_data = np.column_stack((point_list1, point_list2))
    fundamental_mat, point_list1, point_list2 = ransac_model.fit(total_data, N_POINT, RANSAC_THRESH)
    return fundamental_mat, point_list1, point_list2



def extract_features(image1, image2, features, frame):
    if features.get(frame) is not None:
        features1 = features.get(frame).get('features1')
        features2 = features.get(frame).get('features2')
        
        features1 = np.array(features1)
        features2 = np.array(features2)

        ind1 = np.where(features1[:, 1] > CROP_MIN)
        features1 = features1[ind1]
        features2 = features2[ind1]

        ind2 = np.where(features2[:, 1] > CROP_MIN)
        features1 = features1[ind2]
        features2 = features2[ind2]

        features1[:, 1] = features1[:, 1] - CROP_MIN
        features2[:, 1] = features2[:, 1] - CROP_MIN

    else:
        features1 = []
        features2 = []
        '''
        Before using sift:
        1) if opencv_contrib_python is already there and is of some other version (not 3.4.2.16)  
                >> pip uninstall opencv_contrib_python
        
        2) >> python -m pip install --user opencv-contrib-python==3.4.2.16
        '''

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        keypoint1, descriptor1 = sift.detectAndCompute(image1, None)
        keypoint2, descriptor2 = sift.detectAndCompute(image2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptor1, descriptor2, k=2)

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                x1, y1 = keypoint1[m.queryIdx].pt
                x2, y2 = keypoint2[m.trainIdx].pt
                features1.append([x1, y1, 1])
                features2.append([x2, y2, 1])

        write_to_file(features1, features2, frame)

    features1 = np.ascontiguousarray(features1)
    features2 = np.ascontiguousarray(features2)

    return features1, features2


def write_to_file(features1, features2, frame):
    with open(FEATURE_FILE, 'r') as feature_file:
        features = json.load(feature_file)
    
    features[frame] = {
            'features1': list(features1),
            'features2': list(features2)
        }
    with open(FEATURE_FILE, 'w') as feature_file:
        json.dump(features, feature_file)



def get_possible_camera_poses(essential_mat):
    U, _, VT = np.linalg.svd(essential_mat, full_matrices=True)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    #P′ = [UWVT | +u3] or [UWVT | −u3] or [UWTVT | +u3] or [UWTVT | −u3].
    poses = np.array([
        np.column_stack( (U @ W @ VT, U[:, COL3]) ),
        np.column_stack( (U @ W @ VT, -U[:, COL3]) ),
        np.column_stack( (U @ W.T @ VT, U[:, COL3]) ),
        np.column_stack( (U @ W.T @ VT, -U[:, COL3]) )
    ])

    poses = np.array([
        adjust_sign(pose) for pose in poses
    ])
    
    return poses


def adjust_sign(pose):
    pose = -pose if np.linalg.det(pose[:, :COL4]) < 0 else pose
    return pose


def get_correct_pose(poses, point_list1, point_list2, non_linear_flag):
    max_num_positive_depths = 0
    correct_pose = None
    for pose in poses:
        num_positive_depths, better_approx_pose = get_num_positive_depths(pose, point_list1, point_list2, non_linear_flag)
        if num_positive_depths > max_num_positive_depths:
            max_num_positive_depths = num_positive_depths
            correct_pose = better_approx_pose

    return correct_pose


def get_num_positive_depths(pose, point_list1, point_list2, non_linear_flag):
    rot3 = pose[ROW3, :COL4]
    C = pose[:, COL4]
    num_positive_depths = 0
    points_3D = get_linear_triangulated_points(pose, point_list1, point_list2) # N x 3

    ## Non linear triangulation can be done here to get better approximation of the 3D point
    ## points_3D = get_nonlinear_triangulated_points(points_3D, pose, point_list1, point_list2)

    ## We can then have a better approximation of the pose to get R and t
    ## better_approx_pose = get_approx_pose_by_non_linear_pnp(points_3D, pose, point_list1, point_list2)

    better_approx_pose = pose
    if non_linear_flag:
        points_3D = get_nonlinear_triangulated_points(points_3D, pose, point_list1, point_list2)
        better_approx_pose = get_approx_pose_by_non_linear_pnp(np.array(points_3D), pose, point_list1, point_list2)
        rot3 = better_approx_pose[ROW3, :COL4]
        C = better_approx_pose[:, COL4]

    for point in points_3D:
        # cheirality condition
        if (rot3 @ (point - C)) > 0:
            num_positive_depths += 1
    return num_positive_depths, better_approx_pose


def get_linear_triangulated_points(pose, point_list1, point_list2):
    P = np.eye(3,4)
    P_dash = pose
    points_3D = []
    num_points = len(point_list1)
    for i in range(num_points):
        point1 = point_list1[i]
        point2 = point_list2[i]
        '''
        A = np.array([
            (point1[Y] * P[ROW3]) - P[ROW2],
            P[ROW1] - (point1[X]*P[ROW3]),
            (point2[Y] * P_dash[ROW3]) - P_dash[ROW2],
            P_dash[ROW1] - (point2[X] * P_dash[ROW3])
        ])
        '''
        point1_cross = np.array([
            [0, -point1[Z], point1[Y]],
            [point1[Z], 0, -point1[X]],
            [-point1[Y], point1[X], 0]
        ])

        point2_cross = np.array([
            [0, -point2[Z], point2[Y]],
            [point2[Z], 0, -point2[X]],
            [-point2[Y], point2[X], 0]
        ])

        point1_cross_P = point1_cross @ P
        point2_cross_P_dash = point2_cross @ P_dash

        A = np.vstack((point1_cross_P, point2_cross_P_dash))

        _, _, VT = np.linalg.svd(A)
        solution = VT.T[:, -1]
        solution /= solution[-1]

        points_3D.append([solution[X], solution[Y], solution[Z]])
        #yield [solution[X], solution[Y], solution[Z]] 
          
    return points_3D





def get_nonlinear_triangulated_points(points_3D, pose, point_list1, point_list2):
    P = np.eye(3,4)
    P_dash = pose
    tot_points = len(points_3D)
    approx_points = []
    for p1, p2, point_3D in zip(point_list1, point_list2, points_3D):
        x, y, z = point_3D
        est_point = [x, y, z]
        point_3D_approx = least_squares(nlt_error, est_point, args = (P, P_dash, point_list1, point_list2), max_nfev = 10000)
        approx_points.append(point_3D_approx)
    return np.array(approx_points)


def nlt_error(point_3D, P1, P2, points_left, points_right):
    X, Y, Z = point_3D
    point_3D = np.array([X, Y, Z, 1])

    p1 = P1 @ point_3D
    p2 = P2 @ point_3D

    projected_x1, projected_y1 = p1[0]/p1[2], p1[1]/p1[2]
    projected_x2, projected_y2 = p2[0]/p2[2], p2[1]/p2[2]

    dist1 = (points_left[0] - projected_x1)**2 + (points_left[1] - projected_y1)**2
    dist2 = (points_right[0] - projected_x2)**2 + (points_right[1] - projected_y2)**2

    error = dist1 + dist2
    return error




def get_approx_pose_by_non_linear_pnp(points_3D, pose, point_list1, point_list2):
    pose = np.reshape(pose, (-1,1))

    pose_est = [elem for elem in pose]
    print('pose', pose.shape)
    approx_pose = least_squares(non_linear_pnp_error, pose_est, args = (points_3D, point_list1, point_list2), max_nfev = 100000)
    print('\n\n********', approx_pose)
    print('\n\n********', len(approx_pose))
    return np.reshape(approx_pose, (3,4))


def non_linear_pnp_error(pose, points_3D, points_left, points_right):
    P1 = np.eye(3,4)
    P2 = np.reshape(pose, (3,4))

    ## we can have R in the form of R(q), i.e quaternions, to enforce orthogonality for better results

    X, Y, Z = point_3D
    point_3D = np.array([X, Y, Z, 1])

    p1 = P1 @ point_3D
    p2 = P2 @ point_3D

    projected_x1, projected_y1 = p1[0]/p1[2], p1[1]/p1[2]
    projected_x2, projected_y2 = p2[0]/p2[2], p2[1]/p2[2]

    dist1 = (points_left[0] - projected_x1)**2 + (points_left[1] - projected_y1)**2
    dist2 = (points_right[0] - projected_x2)**2 + (points_right[1] - projected_y2)**2

    error = dist1 + dist2
    return error

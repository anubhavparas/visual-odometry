import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


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
    approx_pose = least_squares(non_linear_pnp_error, pose_est, args = (points_3D, point_list1, point_list2), max_nfev = 100000)
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

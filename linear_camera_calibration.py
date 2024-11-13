"""
This code is an implementation of Exercise 1.13 from the book "Computer Vision: A Modern Approach" by Forsyth & Ponce.
"""

import numpy as np


def generate_3d_points(num_points):
    # generate some 3D random points.
    X = np.random.uniform(-1, 1, num_points)
    Y = np.random.uniform(-1, 1, num_points)
    # use different range in order to avoid degenerate configuration
    Z = np.random.uniform(1, 10, num_points)

    # stacks points in one matrix
    points_3D = np.vstack((X, Y, Z)).T

    return points_3D


def create_projection_matrix():
    # intrinsic parameters
    α = 1200
    β = 800
    cx = 320
    cy = 240
    # no manufacturer error
    θ = np.radians(90)

    # extrinsic parameters
    R = np.eye(3)

    t = np.array([[0], [0], [-4]])

    K = np.array([
        [α, α * (np.cos(θ) / np.sin(θ)), cx],
        [0, β / np.sin(θ), cy],
        [0, 0, 1]
    ])

    RT = np.hstack((R, t))
    P = K @ RT

    parameters = {'α': α, 'β': β, 'cx': cx, 'cy': cy, 'θ': θ, 'R': R, 't': t}

    return P, parameters


def project_points(points_3D, P):
    # number of points
    num_points = points_3D.shape[0]
    # set homogeneous coordinates
    points_3D_hom = np.hstack((points_3D, np.ones((num_points, 1))))
    # project points
    points_2D_hom = (P @ points_3D_hom.T).T
    # We must divide by Z to complete the projection x, y =  ((focal_length * scale_factor_x) X / Z, (focal_length *
    # scale_factor_y) * Y / Z)
    # α = (focal_length * scale_factor_x)
    # β = (focal_length * scale_factor_y)# We must divide by Z to complete the projection x, y =  ((focal_length * scale_factor_x) X / Z, (focal_length *
    # scale_factor_y) * Y / Z)
    # α = (focal_length * scale_factor_x)
    # β = (focal_length * scale_factor_y)
    points_2D = points_2D_hom[:, :2] / points_2D_hom[:, 2, np.newaxis]

    return points_2D


def add_noise(points_2D, sigma=1.0):
    noise = np.random.normal(0, sigma, points_2D.shape)
    points_2D_noisy = points_2D + noise

    return points_2D_noisy


if __name__ == "__main__":
    num_points = 100
    points_3D = generate_3d_points(num_points)

    P_i, parameters = create_projection_matrix()

    points_2D = project_points(points_3D, P_i)
    points_2D = add_noise(points_2D, sigma=0.5)

    # P final
    p_final = []

    for idx in range(points_3D.shape[0]):
        # for x points
        p_final.append([
            points_3D[idx][0], points_3D[idx][1], points_3D[idx][2], 1,
            0, 0, 0, 0,
            -points_2D[idx][0] * points_3D[idx][0],
            -points_2D[idx][0] * points_3D[idx][1],
            -points_2D[idx][0] * (points_3D[idx][2]), -points_2D[idx][0]
        ])

        # for y points
        p_final.append([
            0, 0, 0, 0,
            points_3D[idx][0], points_3D[idx][1], (points_3D[idx][2]), 1,
            -points_2D[idx][1] * points_3D[idx][0],
            -points_2D[idx][1] * points_3D[idx][1],
            -points_2D[idx][1] * (points_3D[idx][2]), -points_2D[idx][1]
        ])

    P = np.array(p_final)

    # SVD P, we are looking for the last column of V to minimize (find the best solution in nullspace) ||Pm|| = 0
    # (V, not Vt)
    U, S, Vt = np.linalg.svd(P)
    V = Vt.T

    # last column of V, is the vector m
    m = V[:, -1]

    # reshape m to convert M matrix.
    M = m.reshape(3, 4)

    a1 = M[0, :-1]
    a2 = M[1, :-1]
    a3 = M[2, :-1]

    # get the scale factor
    # we know that a3 has norm=1 (R is orthonormal matrix)
    # we can extract the scale factor using 1 / norm
    p = 1 / np.linalg.norm(a3)

    # α * r1^t  -α*cotθ *r2^T     x0 * r3^T
    #           (β / sinθ) * r2^T y0 * r3^T
    #                r3^T

    # a1 * a3 = x0 * (r3^T)^2
    # (r3^T)² = 1 (dot product with angle 0)
    cx = p ** 2 * (np.dot(a1, a3))  # (pa1 * pa3)
    cy = p ** 2 * (np.dot(a2, a3))

    # radiants
    θ_rad = (np.dot(np.cross(a1, a3), np.cross(a2, a3))) / (
        np.linalg.norm(np.cross(a1, a3) * np.linalg.norm(np.cross(a2, a3))))
    θ_rad = np.arccos(-θ_rad)

    α = np.linalg.norm(np.cross(p * a1, p * a3)) * np.sin(θ_rad)
    β = np.linalg.norm(np.cross(p * a2, p * a3)) * np.sin(θ_rad)

    r3 = p * a3  # a3 -> r3^T
    r1 = (1 / np.linalg.norm(np.cross(a2, a3))) * np.cross(a2, a3)
    r2 = np.cross(r3, r1)

    """
    print("*" * 100)
    print(np.degrees(np.arccos(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))))
    print(np.degrees(np.arccos(np.dot(r1, r3) / (np.linalg.norm(r1) * np.linalg.norm(r3)))))
    print(np.degrees(np.arccos(np.dot(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3)))))
    """

    estimated_matrix_k = np.array([
        [α, α * (np.cos(θ_rad) / np.sin(θ_rad)), cx],
        [0, β / np.sin(θ_rad), cy],
        [0, 0, 1]
    ])

    estimated_rotation_matrix = np.array([
        r1, r2, r3
    ])

    estimated_translation_matrix = (p * (np.linalg.inv(estimated_matrix_k) @ M[:, -1])).reshape(-1, 1)

    RT_final = np.hstack((estimated_rotation_matrix, estimated_translation_matrix))
    P_final = estimated_matrix_k @ RT_final

    print("*" * 100)
    print("ORIGINAL PARAMETERS MATRIX:")
    print(f"parameters: {parameters}")
    print(P_i)
    print("*" * 100)
    print("ESTIMATED PARAMETERS MATRIX:")
    print(P_final)
    print({'α': α, 'β': β, 'cx': cx, 'cy': cy, 'θ_rad': θ_rad, 'R': estimated_rotation_matrix,
           't': estimated_translation_matrix})
    print("*" * 100)
    print("CHECKING POINTS")

    estimated_points = project_points(points_3D, P_final)
    for idx in range(points_2D.shape[0]):
        print(f"ORIGINAL_POINT: {points_2D[idx]} ESTIMATED_POINT{estimated_points[idx]}")

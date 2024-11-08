"""
This code is an implementation of Exercise 1.14 from the book "Computer Vision: A Modern Approach" by Forsyth & Ponce.
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


# rotation vector to rotation matrix
def rodrigues_to_matrix(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        return np.eye(3)
    r = rvec / theta
    K = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return R


def polinomial_lambda_distortion(coe_1, coe_2, coe_3):
    def inner(d):
        # d = (x ** 2) + (y ** 2)
        return (coe_1 * d ** 3) * (coe_2 * d ** 2) + (coe_3 * d) + 1

    return inner


def create_projection_matrix(alpha, beta, cx, cy, theta, coe_r_1, coe_r_2, coe_r_3,
                             r_v_1, r_v_2, r_v_3, t_1, t_2, t_3):
    radial_f = polinomial_lambda_distortion(coe_r_1, coe_r_2, coe_r_3)

    K = np.array([
        [alpha, beta * (np.cos(theta) / np.sin(theta)), cx],
        [0, beta / np.sin(theta), cy],
        [0, 0, 1]
    ])

    R = rodrigues_to_matrix(np.array([r_v_1, r_v_2, r_v_3]))
    t = np.array([[t_1], [t_2], [t_3]])
    Rt = np.hstack((R, t))
    P = K @ Rt

    return P, radial_f, K, R, t, np.array([r_v_1, r_v_2, r_v_3])


def project_points(points_3D, P, K, radial_f):
    # number of points
    num_points = points_3D.shape[0]

    # set homogeneous coordinates 3D WORLD
    points_3D_hom = np.hstack((points_3D, np.ones((num_points, 1))))

    # project points to camera view
    points_2D_hom = (P @ points_3D_hom.T).T

    # We must divide by Z to complete the projection x, y =  ((focal_length * scale_factor_x) X / Z, (focal_length *
    # scale_factor_y) * Y / Z)
    # α = (focal_length * scale_factor_x)
    # β = (focal_length * scale_factor_y)# We must divide by Z to complete the projection x, y =  ((focal_length * scale_factor_x) X / Z, (focal_length *
    # scale_factor_y) * Y / Z)
    # α = (focal_length * scale_factor_x)
    # β = (focal_length * scale_factor_y)
    points_2D = points_2D_hom[:] / points_2D_hom[:, 2, np.newaxis]

    # apply radial distortion
    normalized_points = (np.linalg.inv(K) @ points_2D.T).T

    d_2 = np.linalg.norm(normalized_points, axis=1) - 1
    radial_lambda = radial_f(d_2)

    # points_2D[:, 0] /= radial_lambda
    # points_2D[:, 1] /= radial_lambda

    return points_2D, radial_lambda, normalized_points


def add_noise(points_2D, sigma=1.0):
    noise = np.random.normal(0, sigma, points_2D.shape)
    points_2D_noisy = points_2D + noise

    return points_2D_noisy


def build_P_matrix(points_2D, points_3D):
    # P_y final
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

    return P


def create_skew_rotation_matrix(vector):
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0],
    ])


def compute_jacobian(points_2D_y, points_3D, params):
    # number of points (rows)
    N = points_3D.shape[0]

    # number of params (col)
    M = len(params)

    # create Jacobian matrix
    J = np.zeros((N * 2, M))

    # project points
    P_x, radial_fx, K_x, R_x, t_x, r_v_x = create_projection_matrix(**params)
    points_2D_x, radial_lambda, norm_points = project_points(points_3D, P_x, K_x, radial_fx)
    # flat the indices x,y in one row
    # The jacobian matrix merge x,y values in rows
    # point(x, y) -> x = 2j-1, y = 2j (in the book, index start in 1 not 0)
    residuals = (points_2D_y - points_2D_x).ravel()

    """
    {'alpha': 1200, 'beta': 800, 'cx': 320, 'cy': 240, 'theta': np.radians(90),
     'coe_r_1': 0.005, 'coe_r_2': 0.01, 'coe_r_3': 0.1,
     'r_v_1': 1, 'r_v_2': 1, 'r_v_3': 1,
     't_1': 0, 't_2': 0, 't_3': 3}
     
     K = np.array([
        [alpha, beta * (np.cos(theta) / np.sin(theta)), cx],
        [0, beta / np.sin(theta), cy],
        [0, 0, 1]
    ])
    """

    # x = X/Z, y = Y/Z # normalized coor
    # u = alpha * x * radial + beta * skew_x + cx
    # v = beta * y * radian * skew_y + cy
    # K * ((RX + t) / Z)
    X_CAM = ((R_x @ points_3D.T) + t_x).T

    # d_u/d_alpha, d_v/alpha
    # radial_lambda_xy = np.zeros_like(norm_points)[:, :-1]
    # radial_lambda_xy[:, 0] = norm_points[:, 0] * radial_lambda
    # radial_lambda_xy[:, 1] = norm_points[:, 1] * radial_lambda

    # d_u / d_alpha
    J[::2, 0] = (norm_points[:, 0] * radial_lambda)
    # d_v / d_alpha
    J[1::2, 0] = 0

    # d_u/d_beta, d_v/d_beta
    J[::2, 1] = (norm_points[:, 1] * radial_lambda) * (np.cos(params['theta']) / np.sin(params['theta']))
    J[1::2, 1] = (norm_points[:, 1] * radial_lambda) * (1 / np.sin(params['theta']))

    # d_u/d_cx
    J[::2, 2] = 1
    # d_v/d_cy
    J[1::2, 3] = 1

    # x² + y² = d²
    # lambda is degree 3
    d_2 = np.linalg.norm(norm_points, axis=1) - 1

    # d_u/d_coe_r_1
    J[::2, 5] = ((((params['alpha'] * norm_points[:, 0]) * (polinomial_lambda_distortion(1, 0, 0)(d_2))))
                 + ((norm_points[:, 1] * (
                    np.cos(params['theta']) / np.sin(params['theta'])))) * polinomial_lambda_distortion(1, 0, 0)(
                d_2))
    # d_v/d_coe_r_1
    J[1::2, 5] = (params['beta'] * norm_points[:, 1] * (polinomial_lambda_distortion(1, 0, 0)(d_2))
                  * (1 / np.sin(params['theta'])))

    # d_u/d_coe_r_2
    J[::2, 6] = ((((params['alpha'] * norm_points[:, 0]) * (polinomial_lambda_distortion(0, 1, 0)(d_2))))
                 + ((norm_points[:, 1] * (
                    np.cos(params['theta']) / np.sin(params['theta'])))) * polinomial_lambda_distortion(0, 1, 0)(
                d_2))
    # d_v/d_coe_r_2
    J[1::2, 6] = (norm_points[:, 1] * params['beta'] * (polinomial_lambda_distortion(0, 1, 0)(d_2))
                  * (1 / np.sin(params['theta'])))

    # d_u/d_coe_r_3
    J[::2, 7] = ((((params['alpha'] * norm_points[:, 0]) * (polinomial_lambda_distortion(0, 0, 1)(d_2))))
                 + ((norm_points[:, 1] * (
                    np.cos(params['theta']) / np.sin(params['theta'])))) * polinomial_lambda_distortion(0, 0, 1)(
                d_2))
    # d_v/d_coe_r_3
    J[1::2, 7] = (norm_points[:, 1] * params['beta'] * (polinomial_lambda_distortion(0, 0, 1)(d_2))
                  * (1 / np.sin(params['theta'])))

    # the translation vector is based in u = X/Z and v = Y/Z

    # du/dt1  alpha * radial * ((RX1 + t1) / (RX3 + t3)) + beta * ((RX2 + t2) / (RX3 + t3)) * skew + cx
    # -> (alpha * radial) / Z_CAM
    J[::2, 11] = params['alpha'] * radial_lambda / X_CAM[:, -1]
    # du/dt2  alpha * radial * ((RX1 + t1) / (RX3 + t3)) + beta * ((RX2 + t2) / (RX3 + t3)) * skew + cx
    # -> (alpha * radial) / Z_CAM
    J[::2, 12] = (params['beta'] * radial_lambda * (np.cos(params['theta']) / np.sin(params['theta']))) / X_CAM[:, -1]
    # du/dt3  alpha * radial * ((RX1 + t1) / (RX3 + t3)) + beta * ((RX2 + t2) / (RX3 + t3)) * skew + cx
    # -> -X_CAM[:, 0] / Z_CAM^2
    J[::2, 13] = ((params['alpha'] * radial_lambda * (-X_CAM[:, 0]) / (X_CAM[:, -1] ** 2))
                  + (params['beta'] * radial_lambda * ((np.cos(params['theta']) / np.sin(params['theta'])))
                     * (-X_CAM[:, 1]) / (X_CAM[:, -1] ** 2)))

    # dv/dt1  beta * ((RX2 + t2) / (RX3 + t3)) * skew + cy
    # -> 0
    J[1::2, 11] = 0
    # dv/dt2  beta * ((RX2 + t2) / (RX3 + t3)) * skew + cy
    # -> (alpha * radial * skew) / Z_CAM
    J[1::2, 12] = (params['beta'] * radial_lambda * (1 / np.sin(params['theta']))) / X_CAM[:, -1]
    # dv/dt3  beta * ((RX2 + t2) / (RX3 + t3)) * skew + cy
    # -> -X_CAM[:, 0] / Z_CAM^2
    J[1::2, 13] = (
                (params['beta'] * radial_lambda * (1 / np.sin(params['theta']))) * (-X_CAM[:, 1]) / (X_CAM[:, -1] ** 2))

    # the rotation vector is based in u = X/Z and v = Y/Z

    # Rodriguez angle
    theta = np.linalg.norm(r_v_x)
    # normalization rotation vector
    w = r_v_x / theta

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # [0, -k_z, k_y]
    # [k_z, 0, -k_x]
    # [-k_y, k_x, 0]

    # R = I + sin(theta) * [k]x + (1 - cos(theta)) * ([k]x)²
    # [k]x = r / theta
    # theta -> sqrt(r1² + r2² + r3²)
    # dtheta/dr = r1 / theta
    # dsin_theta/dr = (r1 / theta) * cos(theta)
    # dcos_theta/dr = (r1 / theta) * -sin(theta)
    # dk_dr1 = (1/theta) * ([1,0,0] - r1*r / theta**2)

    # using product rule derivation in Rodrigues
    # sin(theta) * dk/dr + dsin_theta/dr * [k]x +
    # (1 - cos(theta)) * (dk/dr @ [K]x + [K]x @ dk/dr) + (-dcos_theta/dr) * ([k]x)²

    # du/dR1 -> alpha * radial * ((RX1 + t) / (Rx3 + t)) + beta * radial * skew * ((RX2 + t) / (RX3 + t)) + cx

    k_x = r_v_x / theta
    k_x = create_skew_rotation_matrix(k_x)

    # r1
    dtheta_dr1 = r_v_x[0] / theta
    dk_dr1 = (1 / theta) * (np.array([1, 0, 0]) - ((r_v_x * r_v_x[0]) / theta ** 2))
    dk_dr1 = create_skew_rotation_matrix(dk_dr1)
    dsin_dr1 = cos_theta * dtheta_dr1
    dcos_dr1 = -sin_theta * dtheta_dr1

    # dk_r1 k**2 -> k @ k -> product rule -> dk_r1 @ k + k @ dk_r1
    dR_dr1 = ((sin_theta * dk_dr1 + dsin_dr1 * k_x) + ((1 - cos_theta) * (dk_dr1 @ k_x + k_x @ dk_dr1))
              + ((-dcos_dr1) * k_x @ k_x))

    du_dr1_x = (((((dR_dr1 @ points_3D.T).T)[:, 0] + norm_points[:, 2]) - (norm_points[:, 0] *
                                                                           ((dR_dr1 @ points_3D.T).T)[:, 2])) / (
                ((dR_dr1 @ points_3D.T).T)[:, 2]) ** 2) * \
               params["alpha"] * radial_lambda

    du_dr1_y = (((((dR_dr1 @ points_3D.T).T)[:, 1] + norm_points[:, 2]) - (norm_points[:, 1] *
                                                                           ((dR_dr1 @ points_3D.T).T)[:, 2])) / (
                ((dR_dr1 @ points_3D.T).T)[:, 2]) ** 2)

    du_dr1 = du_dr1_x + du_dr1_y
    J[::2, 8] = du_dr1

    # dv/dR1 -> beta * radial * skew * ((RX2 + t) / (RX3 + t)) + cy
    dv_dr1_y = (((((dR_dr1 @ points_3D.T).T)[:, 1] + norm_points[:, 2]) - (norm_points[:, 1] *
                                                                           ((dR_dr1 @ points_3D.T).T)[:, 2])) /
                (((dR_dr1 @ points_3D.T).T)[:, 2]) ** 2)

    dv_dr1 = params['beta'] * radial_lambda * (1 / np.sin(params['theta'])) * dv_dr1_y
    J[1::2, 8] = dv_dr1

    # r2
    dtheta_dr2 = r_v_x[1] / theta
    dk_dr2 = (1 / theta) * (np.array([0, 1, 0]) - ((r_v_x * r_v_x[1]) / theta ** 2))
    dk_dr2 = create_skew_rotation_matrix(dk_dr2)
    dsin_dr2 = cos_theta * dtheta_dr2
    dcos_dr2 = -sin_theta * dtheta_dr2

    # dk_r2 k**2 -> k @ k -> product rule -> dk_r2 @ k + k @ dk_r2
    dR_dr2 = ((sin_theta * dk_dr2 + dsin_dr2 * k_x) + ((1 - cos_theta) * (dk_dr2 @ k_x + k_x @ dk_dr2))
              + ((-dcos_dr2) * k_x @ k_x))

    du_dr2_x = (((((dR_dr2 @ points_3D.T).T)[:, 0] + norm_points[:, 2]) - (norm_points[:, 0] *
                                                                           ((dR_dr2 @ points_3D.T).T)[:, 2])) /
                (((dR_dr2 @ points_3D.T).T)[:, 2]) ** 2) * \
               params["alpha"] * radial_lambda

    du_dr2_y = (((((dR_dr2 @ points_3D.T).T)[:, 1] + norm_points[:, 2]) - (norm_points[:, 1] *
                                                                           ((dR_dr2 @ points_3D.T).T)[:, 2])) /
                (((dR_dr2 @ points_3D.T).T)[:, 2]) ** 2)
    du_dr2 = du_dr2_x + du_dr2_y
    J[::2, 9] = du_dr2

    # dv/dR2 -> beta * radial * skew * ((RX2 + t) / (RX3 + t)) + cy
    dv_dr2_y = (((((dR_dr2 @ points_3D.T).T)[:, 1] + norm_points[:, 2]) - (norm_points[:, 1] *
                                                                           ((dR_dr2 @ points_3D.T).T)[:, 2])) /
                (((dR_dr2 @ points_3D.T).T)[:, 2]) ** 2)

    dv_dr2 = params['beta'] * radial_lambda * (1 / np.sin(params['theta'])) * dv_dr2_y
    J[1::2, 9] = dv_dr2

    # r3
    dtheta_dr3 = r_v_x[2] / theta
    dk_dr3 = (1 / theta) * (np.array([0, 0, 1]) - ((r_v_x * r_v_x[2]) / theta ** 2))
    dk_dr3 = create_skew_rotation_matrix(dk_dr3)
    dsin_dr3 = cos_theta * dtheta_dr3
    dcos_dr3 = -sin_theta * dtheta_dr3

    # dk_r3 k**2 -> k @ k -> product rule -> dk_r3 @ k + k @ dk_r3
    dR_dr3 = ((sin_theta * dk_dr3 + dsin_dr3 * k_x) + ((1 - cos_theta) * (dk_dr3 @ k_x + k_x @ dk_dr3))
              + ((-dcos_dr3) * k_x @ k_x))

    du_dr3_x = (((((dR_dr3 @ points_3D.T).T)[:, 0] + norm_points[:, 2]) - (norm_points[:, 0] *
                                                                           ((dR_dr3 @ points_3D.T).T)[:, 2])) /
                (((dR_dr3 @ points_3D.T).T)[:, 2]) ** 2) * \
               params["alpha"] * radial_lambda

    du_dr3_y = (((((dR_dr3 @ points_3D.T).T)[:, 1] + norm_points[:, 2]) - (norm_points[:, 1] *
                                                                           ((dR_dr3 @ points_3D.T).T)[:, 2])) /
                (((dR_dr3 @ points_3D.T).T)[:, 2]) ** 2)
    du_dr3 = du_dr3_x + du_dr3_y
    J[::2, 10] = du_dr3

    # dv/dR3 -> beta * radial * skew * ((RX2 + t) / (RX3 + t)) + cy
    dv_dr3_y = (((((dR_dr3 @ points_3D.T).T)[:, 1] + norm_points[:, 2]) - (norm_points[:, 1] *
                                                                           ((dR_dr3 @ points_3D.T).T)[:, 2])) /
                (((dR_dr3 @ points_3D.T).T)[:, 2]) ** 2)

    dv_dr3 = params['beta'] * radial_lambda * (1 / np.sin(params['theta'])) * dv_dr3_y
    J[1::2, 10] = dv_dr3

    return J


if __name__ == "__main__":
    num_points = 50
    points_3D = generate_3d_points(num_points)

    # true ground
    parameters_y = {'alpha': 1200, 'beta': 800, 'cx': 320, 'cy': 240, 'theta': np.radians(90),
                    'coe_r_1': 0.005, 'coe_r_2': 0.01, 'coe_r_3': 0.1,
                    'r_v_1': 1, 'r_v_2': 1, 'r_v_3': 1,
                    't_1': 0, 't_2': 0, 't_3': 3}

    P_y, radial_fy, K_y, R_x, t_x, r_v_x = create_projection_matrix(**parameters_y)

    # true ground points_2D
    points_2D_y, radial_lambda_y, _ = project_points(points_3D, P_y, K_y, radial_fy)
    # points_2D_y = add_noise(points_2D_y, sigma=0.5)
    P_y = build_P_matrix(points_2D_y, points_3D)

    # initial point to estimation
    parameters_x = {'alpha': 1200, 'beta': 800, 'cx': 320, 'cy': 240, 'theta': np.radians(90),
                    'coe_r_1': 0.005, 'coe_r_2': 0.01, 'coe_r_3': 0.1,
                    'r_v_1': 1, 'r_v_2': 1, 'r_v_3': 1,
                    't_1': 0, 't_2': 0, 't_3': 3}

    P_x, radial_fx, K_x, R_x, t_x, r_v_x = create_projection_matrix(**parameters_x)

    # pred points_2D
    points_2D_x, radial_lambda_x, _ = project_points(points_3D, P_x, K_x, radial_fx)
    # points_2D_x = add_noise(points_2D_x, sigma=0.5)
    P_x = build_P_matrix(points_2D_x, points_3D)

    compute_jacobian(points_2D_y, points_3D, parameters_x)

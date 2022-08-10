import math

import cv2
import numpy as np


def detect_direction(face_2d, face_3d, img_h, img_w):
    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)

    # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64)

    # The camera matrix
    focal_length = 1 * img_w

    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, _ = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360

    # See where the user's head tilting
    if y < -8:
        return "Left"
    elif y > 8:
        return "Right"
    elif x < -4:
        return "Down"
    elif x > 14:
        return "Up"
    else:
        return "Forward"


def euclidean_dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def validate_blink(eye, threshold=0.19):
    # Finding Distance Right Eye
    hR = euclidean_dist(eye[159], eye[145])
    wR = euclidean_dist(eye[33], eye[133])
    earRight = hR / wR

    # Finding Distance Left Eye
    hL = euclidean_dist(eye[386], eye[374])
    wL = euclidean_dist(eye[263], eye[362])
    earLeft = hL / wL

    ear = (earLeft + earRight) / 2
    if ear < threshold:
        return False
    return True


def validate_smile(mouth, ratio=0.45):
    lips_length = euclidean_dist(mouth[61], mouth[291])
    jaw_length = euclidean_dist(mouth[132], mouth[361])
    return lips_length / jaw_length > ratio


def validate(face_landmarks, img_h, img_w):
    face_3d = []
    face_2d = []
    eye = {}
    mouth = {}
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    for idx, lm in enumerate(face_landmarks.landmark):
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        if min_x == 0:
            min_x = x
        if min_y == 0:
            min_y = y
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
        if idx in [33, 263, 61, 291, 199]:
            # Get the 2D Coordinates
            face_2d.append([x, y])
            # Get the 3D Coordinates
            face_3d.append([x, y, lm.z])
        if idx in [263, 362, 386, 374, 133, 33, 159, 145]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            eye[idx] = (x, y)
        if idx in [61, 291, 132, 361]:
            mouth[idx] = (x, y)

    box = [min_x, min_y, max_x, max_y]
    box[0] = int(box[0] - abs(box[2] - box[0]) * 10 / 100)
    box[1] = int(box[1] - abs(box[3] - box[1]) * 10 / 100)
    box[2] = int(box[2] + abs(box[2] - box[0]) * 10 / 100)
    box[3] = int(box[3] + abs(box[3] - box[1]) * 10 / 100)

    direction = detect_direction(face_2d, face_3d, img_h, img_w)
    blink = validate_blink(eye)
    smile = validate_smile(mouth)

    return box, direction, blink, smile

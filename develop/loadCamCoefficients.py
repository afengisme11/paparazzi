# https://aliyasineser.medium.com/opencv-camera-calibration-e9a48bdd1844
import cv2


def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_matrix = cv_file.getNode("distortion_coefficients").mat()
    cv_file.release()
    return [camera_matrix, dist_matrix]

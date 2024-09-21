import numpy as np
import cv2

# ------------------------------
# ENTER YOUR PARAMETERS HERE:
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
DPI = 300
MARGIN_MM = 10  # margin in millimeters
# ------------------------------

def mm_to_pixels(mm, dpi):
    return int(mm * dpi / 25.4)

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    
    width_pixels = mm_to_pixels(A4_WIDTH_MM - 2 * MARGIN_MM, DPI)
    height_pixels = mm_to_pixels(A4_HEIGHT_MM - 2 * MARGIN_MM, DPI)
    
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    board_height = min(height_pixels, int(width_pixels / size_ratio))
    board_width = int(board_height * size_ratio)
    
    img = cv2.aruco.CharucoBoard.generateImage(board, (board_width, board_height), marginSize=mm_to_pixels(MARGIN_MM, DPI))
    cv2.imshow("img", img)
    cv2.waitKey(2000)
    cv2.imwrite('ChArUco_Marker_A4.png', img)

create_and_save_new_board()
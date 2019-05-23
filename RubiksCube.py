import math

import cv2
import numpy as np
from pprint import pformat
from copy import deepcopy


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x, y)
    # if (event == cv2.EVENT_MOUSEMOVE):
        print("pixel color")
        print(x, y)
        print(param.get(0))
        # frame = cv2.VideoCapture(0).read()
        # cv2.cvtColor(frame)
        # color = frame[x, y]
        # print(cv2.cvtColor(color, cv2.COLOR_BGR2HSV))



def compress_2d_array(original):
    """
    Convert 2d array to a 1d array
    """
    result = []
    for row in original:
        for col in row:
            result.append(col)
    return result


def count_comp(value, list_value):
    c = 0
    for v in list_value:
        if value < v:
            c -= 1
        elif value > v:
            c += 1
    return c


def sort_corners(corner1, corner2, corner3, corner4):
    """
    Sort the corners such that
    - A is top left
    - B is top right
    - C is bottom left
    - D is bottom right

    Return an (A, B, C, D) tuple
    """
    results = []
    corners = (corner1, corner2, corner3, corner4)

    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for (x, y) in corners:
        if min_x is None or x < min_x:
            min_x = x

        if max_x is None or x > max_x:
            max_x = x

        if min_y is None or y < min_y:
            min_y = y

        if max_y is None or y > max_y:
            max_y = y

    # top left
    top_left = None
    top_left_distance = None
    for (x, y) in corners:
        distance = pixel_distance((min_x, min_y), (x, y))
        if top_left_distance is None or distance < top_left_distance:
            top_left = (x, y)
            top_left_distance = distance

    results.append(top_left)

    # top right
    top_right = None
    top_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, min_y), (x, y))
        if top_right_distance is None or distance < top_right_distance:
            top_right = (x, y)
            top_right_distance = distance
    results.append(top_right)

    # bottom left
    bottom_left = None
    bottom_left_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((min_x, max_y), (x, y))

        if bottom_left_distance is None or distance < bottom_left_distance:
            bottom_left = (x, y)
            bottom_left_distance = distance
    results.append(bottom_left)

    # bottom right
    bottom_right = None
    bottom_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, max_y), (x, y))

        if bottom_right_distance is None or distance < bottom_right_distance:
            bottom_right = (x, y)
            bottom_right_distance = distance
    results.append(bottom_right)

    return results


def square_width_height(approx, debug):
    """
    This assumes that approx is a square. Return the width and height of the square.
    """
    width = 0
    height = 0

    # Find the four corners
    (A, B, C, D) = sort_corners(tuple(approx[0][0]),
                                tuple(approx[1][0]),
                                tuple(approx[2][0]),
                                tuple(approx[3][0]))

    # Find the lengths of all four sides
    AB = pixel_distance(A, B)
    AC = pixel_distance(A, C)
    DB = pixel_distance(D, B)
    DC = pixel_distance(D, C)

    width = max(AB, DC)
    height = max(AC, DB)

    if debug:
        pass
        # log.info(
        #     "square_width_height: AB %d, AC %d, DB %d, DC %d, width %d, height %d" % (AB, AC, DB, DC, width, height))

    return width, height


def pixel_distance(a, b):
    (col_a, row_a) = a
    (col_b, row_b) = b

    return math.sqrt(math.pow(col_b - col_a, 2) + math.pow(row_b - row_a, 2))


def get_angle(A, B, C):
    """
    Return the angle at C (in radians) for the triangle formed by A, B, C
    a, b, c are lengths

        C
       / \
    b /   \a
     /     \
    A-------B
        c

    """
    (col_A, row_A) = A
    (col_B, row_B) = B
    (col_C, row_C) = C
    a = pixel_distance(C, B)
    b = pixel_distance(A, C)
    c = pixel_distance(A, B)

    try:
        cos_angle = (math.pow(a, 2) + math.pow(b, 2) - math.pow(c, 2)) / (2 * a * b)
    except ZeroDivisionError as e:
        # log.warning("get_angle: A %s, B %s, C %s, a %.3f, b %.3f, c %.3f" % (A, B, C, a, b, c))
        raise e

    # If CA and CB are very long and the angle at C very narrow we can get an
    # invalid cos_angle which will cause math.acos() to raise a ValueError exception
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1

    angle_ACB = math.acos(cos_angle)
    # log.info("get_angle: A %s, B %s, C %s, a %.3f, b %.3f, c %.3f, cos_angle %s, angle_ACB %s" %
    #          (A, B, C, a, b, c, pformat(cos_angle), int(math.degrees(angle_ACB))))
    return angle_ACB


class CustomContour(object):

    def __init__(self, rubiks_parent, index, contour, hierarchy, debug):
        self.rubiks_parent = rubiks_parent
        self.index = index
        self.contour = contour
        self.hierarchy = hierarchy
        peri = cv2.arcLength(contour, True)
        self.approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        self.area = cv2.contourArea(contour)
        self.corners = len(self.approx)
        self.width = None
        self.debug = debug

        # compute the center of the contour
        m = cv2.moments(contour)

        if m["m00"]:
            self.cX = int(m["m10"] / m["m00"])
            self.cY = int(m["m01"] / m["m00"])
        else:
            self.cX = None
            self.cY = None

    def approx_is_square(self, side_vs_side_threshold=0.60, angle_threshold=20, rotate_threshold=30):
        """
        Rules
        - there must be four corners
        - all four lines must be roughly the same length
        - all four corners must be roughly 90 degrees
        - AB and CD must be horizontal lines
        - AC and BC must be vertical lines

        SIDE_VS_SIDE_THRESHOLD
            If this is 1 then all 4 sides must be the exact same length.  If it is
            less than one that all sides must be within the percentage length of
            the longest side.

        ANGLE_THRESHOLD
            If this is 0 then all 4 corners must be exactly 90 degrees.  If it
            is 10 then all four corners must be between 80 and 100 degrees.

        ROTATE_THRESHOLD
            Controls how many degrees the entire square can be rotated

        The corners are labeled

            A ---- B
            |      |
            |      |
            C ---- D
        """

        assert 0 <= side_vs_side_threshold <= 1, "SIDE_VS_SIDE_THRESHOLD must be between 0 and 1"
        assert 0 <= angle_threshold <= 90, "ANGLE_THRESHOLD must be between 0 and 90"

        # Проверяем что углов 4
        if len(self.approx) != 4:
            return False

        # Find the four corners
        (a, b, c, d) = sort_corners(tuple(self.approx[0][0]),
                                    tuple(self.approx[1][0]),
                                    tuple(self.approx[2][0]),
                                    tuple(self.approx[3][0]))

        # Find the lengths of all four sides
        ab = pixel_distance(a, b)
        ac = pixel_distance(a, c)
        db = pixel_distance(d, b)
        dc = pixel_distance(d, c)
        distances = (ab, ac, db, dc)
        max_distance = max(distances)
        cutoff = int(max_distance * side_vs_side_threshold)

        # If any side is much smaller than the longest side, return False
        for distance in distances:
            if distance < cutoff:
                return False

        # all four corners must be roughly 90 degrees
        min_angle = 90 - angle_threshold
        max_angle = 90 + angle_threshold

        # Angle at A
        angle_A = int(math.degrees(get_angle(c, b, a)))
        if angle_A < min_angle or angle_A > max_angle:
            return False

        # Angle at B
        angle_B = int(math.degrees(get_angle(a, d, b)))
        if angle_B < min_angle or angle_B > max_angle:
            return False

        # Angle at C
        angle_C = int(math.degrees(get_angle(a, d, c)))
        if angle_C < min_angle or angle_C > max_angle:
            return False

        # Angle at D
        angle_D = int(math.degrees(get_angle(c, b, d)))
        if angle_D < min_angle or angle_D > max_angle:
            return False

        far_left = min(a[0], b[0], c[0], d[0])
        far_right = max(a[0], b[0], c[0], d[0])
        far_up = min(a[1], b[1], c[1], d[1])
        far_down = max(a[1], b[1], c[1], d[1])
        top_left = (far_left, far_up)
        top_right = (far_right, far_up)
        bottom_left = (far_left, far_down)
        bottom_right = (far_right, far_down)
        debug = False

        '''
        if A[0] >= 93 and A[0] <= 96 and A[1] >= 70 and A[1] <= 80:
            debug = True
            log.info("approx_is_square A %s, B, %s, C %s, D %s, distance AB %d, AC %d, DB %d, DC %d, max %d, cutoff %d, angle_A %s, angle_B %s, angle_C %s, angle_D %s, top_left %s, top_right %s, bottom_left %s, bottom_right %s" %
                     (A, B, C, D, AB, AC, DB, DC, max_distance, cutoff,
                      angle_A, angle_B, angle_C, angle_D,
                      pformat(top_left), pformat(top_right), pformat(bottom_left), pformat(bottom_right)))
        '''

        # Is AB horizontal?
        if b[1] < a[1]:
            # Angle at B relative to the AB line
            angle_B = int(math.degrees(get_angle(a, top_left, b)))

            if debug:
                pass
                # log.info("AB is horizontal, angle_B %s, ROTATE_THRESHOLD %s" % (angle_B, rotate_threshold))

            if angle_B > rotate_threshold:
                if debug:
                    pass
                    # log.info("AB horizontal rotation %s is above ROTATE_THRESHOLD %s" % (angle_B, rotate_threshold))
                return False
        else:
            # Angle at A relative to the AB line
            angle_A = int(math.degrees(get_angle(b, top_right, a)))

            if debug:
                pass
                # log.info("AB is vertical, angle_A %s, ROTATE_THRESHOLD %s" % (angle_A, rotate_threshold))

            if angle_A > rotate_threshold:
                if debug:
                    pass
                    # log.info("AB vertical rotation %s is above ROTATE_THRESHOLD %s" % (angle_A, rotate_threshold))
                return False

        # TODO - if the area of the approx is way more than the
        # area of the contour then this is not a square
        return True

    def is_square(self, target_area=None):
        AREA_THRESHOLD = 0.50

        if not self.approx_is_square():
            return False

        if self.width is None:
            (self.width, self.height) = square_width_height(self.approx, self.debug)

        if target_area:
            area_ratio = float(target_area / self.area)

            if area_ratio < float(1.0 - AREA_THRESHOLD) or area_ratio > float(1.0 + AREA_THRESHOLD):
                # log.info("FALSE %s target_area %d, my area %d, ratio %s" % (self, target_area, self.area, area_ratio))
                return False
            else:
                # log.info("TRUE %s target_area %d, my area %d, ratio %s" % (self, target_area, self.area, area_ratio))
                return True
        else:
            return True

    def get_child(self):
        # Each contour has its own information regarding what hierarchy it
        # is, who is its parent, who is its parent etc. OpenCV represents it as
        # an array of four values : [Next, Previous, First_child, Parent]
        child = self.hierarchy[2]

        if child == -1:
            return None
        else:
            return self.rubiks_parent.contours_by_index[child]

    def __str__(self):
        return "Contour #%d (%s, %s)" % (self.index, self.cX, self.cY)


class RubiksCube:

    def __init__(self, debug=False):
        self.data = {}
        self.debug = debug
        self.candidates = []
        self.width = None
        self.height = None
        self.raw_contours = None
        self.hierarchy = None
        self.contours_by_index = {}
        self.mean_square_area = 0
        self.median_square_area = 0
        self.median_square_width = 0
        self.size = 3
        self.cubeImgSize = 42
        self.color_dict = {"WHITE": (255, 255, 255),
                           "GREEN": (0, 255, 0),
                           "BLUE": (255, 0, 0),
                           "RED": (0, 0, 255),
                           "YELLOW": (0, 255, 255),
                           "ORANGE": (0, 123, 255),
                           "BLACK": (0, 0, 0)}
        self.color_nums = {
                           "ORANGE": 0,
                           "GREEN": 1,
                           "WHITE": 2,
                           "BLUE": 3,
                           "RED": 4,
                           "YELLOW": 5}
        self.side_colors = []
        self.side_coordinates = []
        self.init_side_coords()
        self.init_side_colors()

    def display(self, image, desc, missing=None):
        if not self.debug:
            return
        if missing is None:
            missing = []

        if self.candidates:
            to_draw = []
            to_draw_square = []
            to_draw_approx = []
            to_draw_missing = []
            to_draw_missing_approx = []

            for con in missing:
                to_draw_missing.append(con.contour)
                to_draw_missing_approx.append(con.approx)

            for con in self.candidates:
                if con in missing:
                    continue

                if con.is_square():
                    to_draw_square.append(con.contour)
                    to_draw_approx.append(con.approx)
                else:
                    to_draw.append(con.contour)
                    to_draw_approx.append(con.approx)

            tmp_image = image.copy()
            # cons that are squares are in green
            # for non-squqres the approx is green and contour is blue
            cv2.drawContours(tmp_image, to_draw, -1, (255, 0, 0), 2)
            cv2.drawContours(tmp_image, to_draw_approx, -1, (0, 0, 255), 2)
            cv2.drawContours(tmp_image, to_draw_square, -1, (0, 255, 0), 2)

            # if to_draw_missing:
            #     cv2.drawContours(tmp_image, to_draw_missing, -1, (0, 255, 255), 2)
            #     cv2.drawContours(tmp_image, to_draw_missing_approx, -1, (255, 255, 0), 2)

            cv2.imshow(desc, tmp_image)
            cv2.setMouseCallback(desc, click_event)
            cv2.waitKey(0)
        else:
            cv2.imshow(desc, image)
            cv2.setMouseCallback(desc, click_event)
            cv2.waitKey(0)
            """
            d = []
            for c in self.candidates:
                d.append(c.contour)
            image = cv2.drawContours(image, d, -1, (255, 0, 0), 2)
            for c in self.candidates:
                if c.cX is None:
                    continue
                cv2.addText(image, str(c.area), (c.cX, c.cY), "")

            cv2.imshow("areas", image)
            cv2.setMouseCallback("areas", click_event)
            cv2.waitKey(0)"""

    def analyze_file(self, file):
        image = cv2.imread(file)
        self.analyze(image)

    def reset(self):
        self.data = {}
        self.candidates = []
        self.contours_by_index = {}
        self.mean_square_area = None
        self.median_square_area = None
        self.black_border_width = None
        self.top = None
        self.right = None
        self.bottom = None
        self.left = None
        self.mean_square_area = 0
        self.median_square_area = 0
        self.median_square_width = 0

        # 2 for 2x2x2, 3 for 3x3x3, etc
        self.size = 3
        # self.configuration = []


    def analyze_side(self, frame):
        width = 20
        count = 0;
        con_color = []
        for (con, hsv) in zip(self.sort_by_row_col(deepcopy(self.candidates), self.size), self.data.values()):
            if con.width:
                # hsv = cv2.cvtColor(np.uint8([[(color[2], color[1], color[0])]]), cv2.COLOR_BGR2HSV)[0][0]
                # print(hsv)
                # minHSV = np.array([98, 109, 20])
                # maxHSV = np.array([130, 255, 255])

                # print(minHSV)
                # print(maxHSV)

                # if (0 <= hsv[0] <= 5 and 140 <= hsv[1] <= 255 and 140 <= hsv[2] <= 255) or \
                #      (140 <= hsv[0] <= 180 and 140 <= hsv[1] <= 255 and 50 <= hsv[2] <= 255):
                #     print("must be red")
                # print(hsv)
                print("++++++++")
                # print(cv2.inRange(np.uint8([[hsv]]), minHSV, maxHSV))

                # white
                if 0 <= hsv[1] <= 30 and 150 <= hsv[2] <= 255:
                    color = self.color_dict['WHITE'] #(255, 255, 255)
                # blue
                elif 98 <= hsv[0] <= 130 and 109 <= hsv[1] <= 255 and 20 <= hsv[2] <= 255:
                    color = self.color_dict['BLUE'] #(255, 0, 0)
                # orange
                elif 2 <= hsv[0] <= 19 and 155 <= hsv[1] <= 211 and 190 <= hsv[2] <= 235:
                    color = self.color_dict['ORANGE'] #(0, 123, 255)
                # red
                elif (0 <= hsv[0] <= 5 and 140 <= hsv[1] <= 255 and 140 <= hsv[2] <= 255) or \
                        (140 <= hsv[0] <= 180 and 140 <= hsv[1] <= 255 and 50 <= hsv[2] <= 255):
                    # print("must be red")
                    # print(hsv)
                    color = self.color_dict['RED'] #(0, 0, 255)
                # yellow
                elif 20 <= hsv[0] <= 40 and 100 <= hsv[1] <= 255 and 20 <= hsv[2] <= 255:
                    color = self.color_dict['YELLOW'] #(0, 255, 255)
                # green
                elif 57 <= hsv[0] <= 80 and 100 <= hsv[1] <= 150 and 120 <= hsv[2] <= 210:
                    color = self.color_dict['GREEN'] #(0, 255, 0)
                # black
                else:
                    # print("unknown:")
                    # print(hsv)
                    color = self.color_dict['BLACK'] #(0, 0, 0)
                cv2.circle(frame,
                           (con.cX, con.cY),
                           # int(con.width / 2),
                           width,
                           color,
                           2)
                con_color.append(color)
                count = count + 1
        # print(count)
        # print("con color")
        # print(con_color)
        return count, con_color

    def init_side_coords(self):
        cubeImgSize = self.cubeImgSize
        gap = self.cubeImgSize // 4
        startX = 10
        startY = cubeImgSize * 6
        self.side_coordinates.insert(self.color_nums['ORANGE'], (startX + (cubeImgSize + gap)*0, startY + (cubeImgSize + gap)*1))
        self.side_coordinates.insert(self.color_nums['GREEN'], (startX + (cubeImgSize + gap)*1, startY + (cubeImgSize + gap)*0))
        self.side_coordinates.insert(self.color_nums['WHITE'], (startX + (cubeImgSize + gap)*1, startY + (cubeImgSize + gap)*1))
        self.side_coordinates.insert(self.color_nums['BLUE'], (startX + (cubeImgSize + gap)*1, startY + (cubeImgSize + gap)*2))
        self.side_coordinates.insert(self.color_nums['RED'], (startX + (cubeImgSize + gap)*2, startY + (cubeImgSize + gap)*1))
        self.side_coordinates.insert(self.color_nums['YELLOW'], (startX + (cubeImgSize + gap)*3, startY + (cubeImgSize + gap)*1))

    def init_side_colors(self):
        con_color = []
        for i in range(0, 9):
            con_color.insert(i, (0, 0, 0))
        for i in range(0, 6):
            self.side_colors.insert(i, con_color)

    def paint_cube_sides(self, frame, con_color):
        center_index = 4
        if con_color[center_index] == self.color_dict['ORANGE'] :
            # startX += (cubeImgSize + gap)*0
            # startY += (cubeImgSize + gap)*1
            self.side_colors[self.color_nums['ORANGE']] =  con_color
        elif con_color[center_index] == self.color_dict['GREEN'] :
            # startX += (cubeImgSize + gap)*1
            # startY += (cubeImgSize + gap)*0
            self.side_colors[self.color_nums['GREEN']] = con_color
        elif con_color[center_index] == self.color_dict['WHITE'] :
            # startX += (cubeImgSize + gap)*1
            # startY += (cubeImgSize + gap)*1
            self.side_colors[self.color_nums['WHITE']] = con_color
        elif con_color[center_index] == self.color_dict['BLUE'] :
            # startX += (cubeImgSize + gap)*1
            # startY += (cubeImgSize + gap)*2
            self.side_colors[self.color_nums['BLUE']] = con_color
        elif con_color[center_index] == self.color_dict['RED']:
            # startX += (cubeImgSize + gap) * 2
            # startY += (cubeImgSize + gap) * 1
            self.side_colors[self.color_nums['RED']] = con_color
        elif con_color[center_index] == self.color_dict['YELLOW']:
            # startX += (cubeImgSize + gap) * 3
            # startY += (cubeImgSize + gap) * 1
            self.side_colors[self.color_nums['YELLOW']] = con_color

        cubeImgSize = self.cubeImgSize //3
        gap = self.cubeImgSize // 8
        for sideIndex in range(0, 6):
            for i in range(0, 9):
                localStartX = self.side_coordinates[sideIndex][0] + cubeImgSize * (i % self.size) + gap * (i % self.size)
                localStartY = self.side_coordinates[sideIndex][1] + cubeImgSize * (i // self.size) + gap * (i // self.size)
                first_voord = (localStartX, localStartY)
                second_coord = (localStartX + cubeImgSize, localStartY + cubeImgSize)
                cv2.rectangle(frame, first_voord, second_coord, self.side_colors[sideIndex][i], 3)


    def analyze_video(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('frame')
        self.reset()

        # сделать сохранение кубиков, чтобы не "лагало"
        con_color = []
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.setMouseCallback('frame', click_event, self.data.g)
            cv2.putText(frame, 'halo ma frend', (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            try:
                if not self.analyze(frame):
                    continue
                count, con_color = self.analyze_side(frame)
                if count != self.size*self.size:
                    continue;

            except Exception as _:
                pass

            if len(con_color) == self.size*self.size:
                self.paint_cube_sides(frame, con_color)


            cv2.imshow('frame', frame)
            self.reset()
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def remove_candidate_contour(self, contour_to_remove):

        # hierarchy is [Next, Previous, First_child, Parent]
        (_, _, contour_to_remove_child, contour_to_remove_parent) = contour_to_remove.hierarchy
        # log.warning("removing %s with child %s, parent %s" % (contour_to_remove, contour_to_remove_child, contour_to_remove_parent))

        for con in self.candidates:
            if con == contour_to_remove:
                continue

            (_, _, child, parent) = con.hierarchy
            # log.info("    %s child %s, parent %s" % (con, child, parent))

            if child == contour_to_remove.index:
                con.hierarchy[2] = contour_to_remove_child
                # log.info("    %s child is now %s" % (con, con.hierarchy[2]))

            if parent == contour_to_remove.index:
                con.hierarchy[3] = contour_to_remove_parent

                if contour_to_remove_parent != -1:
                    self.contours_by_index[contour_to_remove_parent].hierarchy[2] = con.index

                # log.info("    %s parent is now %s" % (con, con.hierarchy[3]))

        del self.contours_by_index[contour_to_remove.index]
        self.candidates.remove(contour_to_remove)

        for con in self.candidates:
            child_index = con.hierarchy[2]
            parent_index = con.hierarchy[3]

            if child_index != -1:
                child = self.contours_by_index[child_index]
                child.hierarchy[3] = con.index

            if parent_index != -1:
                parent = self.contours_by_index[parent_index]
                parent.hierarchy[2] = con.index

    def remove_non_square_candidates(self, target_square_area=None):
        """
        Remove non-square contours from candidates.  Return a list of the ones we removed.
        """
        candidates_to_remove = []

        for con in self.candidates:
            if not con.is_square(target_square_area):
                candidates_to_remove.append(con)

        for x in candidates_to_remove:
            self.remove_candidate_contour(x)

        removed = len(candidates_to_remove)

        if self.debug:
            pass
            # log.info("remove-non-square-candidates: %d removed, %d remain, target_square_area %s\n" %
            #          (removed, len(self.candidates), target_square_area))

        if removed:
            return True
        else:
            return False

    def remove_gigantic_candidates(self, area_cutoff):
        candidates_to_remove = []

        # Remove parents with square child contours
        for con in self.candidates:
            if con.area > area_cutoff:
                candidates_to_remove.append(con)
                if self.debug:
                    pass
                    # log.info("remove_gigantic_candidates: %s area %d is greater than cutoff %d" % (
                    # con, con.area, area_cutoff))

        for x in candidates_to_remove:
            self.remove_candidate_contour(x)

        removed = len(candidates_to_remove)

        if self.debug:
            pass
            # log.info("remove-gigantic-candidates: %d removed, %d remain\n" % (removed, len(self.candidates)))

        return candidates_to_remove

    def get_median_square_area(self):
        """
        Find the median area of all square contours
        """
        square_areas = []
        square_widths = []
        total_square_area = 0

        for con in self.candidates:
            if con.is_square():
                square_areas.append(int(con.area))
                square_widths.append(int(con.width))
                total_square_area += int(con.area)

        if square_areas:
            square_areas = sorted(square_areas)
            square_widths = sorted(square_widths)
            num_squares = len(square_areas)

            # Do not take the exact median, take the one 2/3 of the way through
            # the list. Sometimes you get clusters of smaller squares which can
            # throw us off if we take the exact median.
            square_area_index = int((2 * num_squares) / 3)

            self.mean_square_area = int(total_square_area / len(square_areas))
            self.median_square_area = int(square_areas[square_area_index])
            self.median_square_width = int(square_widths[square_area_index])

            if self.debug:
                pass

        else:
            self.mean_square_area = 0
            self.median_square_area = 0
            self.median_square_width = 0

        if not square_areas:
            # raise CubeNotFound("%s no squares in image" % self.name)
            raise Exception("%s no squares in image" % self)

    def remove_dwarf_candidates(self, area_cutoff):
        candidates_to_remove = []

        # Remove parents with square child contours
        for con in self.candidates:
            if con.area < area_cutoff:
                candidates_to_remove.append(con)

        for x in candidates_to_remove:
            self.remove_candidate_contour(x)

        removed = len(candidates_to_remove)

        if self.debug:
            # log.info("remove-dwarf-candidates %d removed, %d remain\n" % (removed, len(self.candidates)))
            pass

        return candidates_to_remove

    def remove_square_within_square_candidates(self):
        candidates_to_remove = []

        # All non-square contours have been removed by this point so remove any contour
        # that has a child contour. This allows us to remove the outer square in the
        # "square within a square" scenario.
        for con in self.candidates:
            child = con.get_child()

            if child:
                candidates_to_remove.append(con)

                if self.debug:
                    pass
                    # log.info("con %s will be removed, has child %s" % (con, child))
            else:
                if self.debug:
                    pass
                    # log.info("con %s will remain" % con)

        for x in candidates_to_remove:
            self.remove_candidate_contour(x)

        removed = len(candidates_to_remove)

        if self.debug:
            pass
            # log.info("remove-square-within-square-candidates %d removed, %d remain\n" % (removed, len(self.candidates)))

        return True if removed else False

    def get_contour_neighbors(self, contours, target_con):
        """
        Return stats on how many other contours are in the same 'row' or 'col' as target_con

        TODO: This only works if the cube isn't at an angle...would be cool to work all the time
        """
        row_neighbors = 0
        row_square_neighbors = 0
        col_neighbors = 0
        col_square_neighbors = 0

        # Wiggle +/- 50% the median_square_width
        if self.size is None:
            WIGGLE_THRESHOLD = 0.50

        elif self.size == 7:
            WIGGLE_THRESHOLD = 0.50

        else:
            WIGGLE_THRESHOLD = 0.70

        # width_wiggle determines how far left/right we look for other contours
        # height_wiggle determines how far up/down we look for other contours
        width_wiggle = int(self.median_square_width * WIGGLE_THRESHOLD)
        height_wiggle = int(self.median_square_width * WIGGLE_THRESHOLD)

        # log.debug("get_contour_neighbors() for %s, median square width %s, width_wiggle %s, height_wiggle %s" %
        #           (target_con, self.median_square_width, width_wiggle, height_wiggle))

        for con in contours:

            # do not count yourself
            if con == target_con:
                continue

            x_delta = abs(con.cX - target_con.cX)
            y_delta = abs(con.cY - target_con.cY)

            if x_delta <= width_wiggle:
                col_neighbors += 1

                if con.is_square(self.mean_square_area):
                    col_square_neighbors += 1
                    # log.debug("%s is a square col neighbor" % con)
                else:
                    pass
                    # log.debug("%s is a non-square col neighbor, it has %d corners" % (con, con.corners))
            else:
                pass
                # log.debug("%s x delta %s is outside width wiggle room %s" % (con, x_delta, width_wiggle))

            if y_delta <= height_wiggle:
                row_neighbors += 1

                if con.is_square(self.mean_square_area):
                    row_square_neighbors += 1
                    # log.debug("%s is a square row neighbor" % con)
                else:
                    pass
                    # log.debug("%s is a non-square row neighbor, it has %d corners" % (con, con.corners))
            else:
                pass
                # log.debug("%s y delta %s is outside height wiggle room %s" % (con, y_delta, height_wiggle))

        # log.debug("get_contour_neighbors() for %s has row %d, row_square %d, col %d, col_square %d neighbors\n" %
        #    (target_con, row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors))

        return row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors

    def get_cube_boundary(self, strict):
        """
        Find the top, right, bottom, left boundry of all square contours
        """
        self.top = None
        self.right = None
        self.bottom = None
        self.left = None

        for con in self.candidates:
            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) = \
                self.get_contour_neighbors(self.candidates, con)

            if self.debug:
                pass
                # log.info("get_cube_boundry: %s row_neighbors %s, col_neighbors %s" %
                #          (con, row_square_neighbors, col_square_neighbors))

            if strict:
                if self.size == 7:
                    if row_square_neighbors < 2 or col_square_neighbors < 2:
                        continue

                elif self.size == 6:
                    if row_square_neighbors < 2 or col_square_neighbors < 2:
                        continue

                elif self.size == 5:
                    if row_square_neighbors < 1 or col_square_neighbors < 1:
                        continue

                elif self.size == 4:
                    if row_square_neighbors < 1 or col_square_neighbors < 1:
                        continue

                elif self.size == 2:
                    if not row_neighbors and not col_neighbors:
                        continue
                else:
                    if not row_neighbors and not col_neighbors:
                        continue

                    if not col_neighbors and row_neighbors >= self.size:
                        continue

                    if not row_neighbors and col_neighbors >= self.size:
                        continue
            else:
                # Ignore the rogue square with no neighbors. I used to do an "or" here but
                # that was too strict for 2x2x2 cubes.
                if not row_square_neighbors and not col_square_neighbors:
                    continue

            if self.top is None or con.cY < self.top:
                self.top = con.cY

            if self.bottom is None or con.cY > self.bottom:
                self.bottom = con.cY

            if self.left is None or con.cX < self.left:
                self.left = con.cX

            if self.right is None or con.cX > self.right:
                self.right = con.cX

        if self.size:
            if self.debug:
                pass
                # log.info("get_cube_boundry: size %s, strict %s, top %s, bottom %s, left %s right %s" %
                #          (self.size, strict, self.top, self.bottom, self.left, self.right))

        if self.top:
            return True
        else:
            return False

    def get_cube_size(self):
        """
        Look at all of the contours that are squares and see how many square
        neighbors they have in their row and col. Store the number of square
        contours in each row/col in data, then sort data and return the
        median entry
        """

        size_count = {}
        self.size = None

        for con in self.candidates:
            if con.is_square(self.median_square_area):
                (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) = \
                    self.get_contour_neighbors(self.candidates, con)
                row_size = row_neighbors + 1
                col_size = col_neighbors + 1
                # log.debug("%s has %d row size, %d col size" % (con, row_size, col_size))

                if row_size not in size_count:
                    size_count[row_size] = 0
                size_count[row_size] += 1

                if col_size not in size_count:
                    size_count[col_size] = 0
                size_count[col_size] += 1

        # Find the size count entry with the highest value, that is our cube size
        cube_size = None
        cube_size_count = 0

        for size in sorted(size_count.keys()):
            count = size_count[size]

            if cube_size is None or count > cube_size_count or (count == cube_size_count and size > cube_size):
                cube_size = size
                cube_size_count = count

        if cube_size == 1:
            self.size = None
        else:

            # self.size_static is None by default and is only set when running in --webcam mode
            if self.size_static is None or cube_size == self.size_static:
                self.size = cube_size

        if self.debug:
            pass
            # log.info("cube size is %s, size_count %s" % (self.size, pformat(size_count)))

        # Return True if we found a valid cube size
        return self.size in (2, 3, 4, 5, 6, 7, 8, 9, 10)

    def set_contour_row_col_index(self, con):
        cube_height = self.bottom - self.top + self.median_square_width
        cube_width = self.right - self.left + self.median_square_width

        # row_size is how many rows of squares we see
        row_size = round(cube_height / float(self.median_square_width + self.black_border_width))
        row_size = min(self.size, row_size)
        median_row_height = float(cube_height / row_size)

        # col_size is how many cols of squares we see
        col_size = round(cube_width / float(self.median_square_width + self.black_border_width))
        col_size = min(self.size, col_size)
        median_col_width = float(cube_width / col_size)

        if self.debug:
            pass
            # log.info("border width %s" % self.black_border_width)
            # log.info(
            #     "set_contour_row_col_index %s, top %s, bottom %s, cube_height %s, row_size %s, median_row_height %s, median_square_width %s" %
            #     (con, self.top, self.bottom, cube_height, row_size, median_row_height, self.median_square_width))
            # log.info(
            #     "set_contour_row_col_index %s, left %s, right %s, cube_width %s, col_size %s, median_col_width %s, median_square_width %s" %
            #     (con, self.left, self.right, cube_width, col_size, median_col_width, self.median_square_width))

        con.row_index = int(round((con.cY - self.top) / median_row_height))
        con.col_index = int(round((con.cX - self.left) / median_col_width))

        if con.row_index >= self.size:
            # raise RowColSizeMisMatch("con.row_index is %s, must be less than size %s" % (con.row_index, self.size))
            raise Exception("con.row_index is %s, must be less than size %s" % (con.row_index, self.size))

        if con.col_index >= self.size:
            # raise RowColSizeMisMatch("con.col_index is %s, must be less than size %s" % (con.col_index, self.size))
            raise Exception("con.col_index is %s, must be less than size %s" % (con.col_index, self.size))

        if self.debug:
            pass
            # log.info("set_contour_row_col_index %s, col_index %s, row_index %s\n" % (con, con.col_index, con.row_index))

    def remove_contours_outside_cube(self, contours):
        assert self.median_square_area is not None, "get_median_square_area() must be called first"
        contours_to_remove = []

        # Give a little wiggle room
        # TODO...I think this should be /2 not /4 because self.top is the center
        # point of the contour closest to the top of the image, it isn't the top of that contour
        top = self.top - int(self.median_square_width / 4)
        bottom = self.bottom + int(self.median_square_width / 4)
        left = self.left - int(self.median_square_width / 4)
        right = self.right + int(self.median_square_width / 4)

        for con in contours:
            if con.cY < top or con.cY > bottom or con.cX < left or con.cX > right:
                contours_to_remove.append(con)

        for x in contours_to_remove:
            self.remove_candidate_contour(x)

        removed = len(contours_to_remove)
        # log.debug("remove-contours-outside-cube %d removed, %d remain" % (removed, len(contours)))
        return True if removed else False

    def get_black_border_width(self):
        cube_height = self.bottom - self.top
        cube_width = self.right - self.left

        if cube_width <= cube_height:
            pixels = cube_width
            pixels_desc = 'width'
        else:
            pixels = cube_height
            pixels_desc = 'height'

        # subtract the pixels of the size-1 squares
        pixels -= (self.size - 1) * self.median_square_width
        self.black_border_width = int(pixels / (self.size - 1))

        if self.debug:
            pass
            # log.warning(
            #     "get_black_border_width: top %s, bottom %s, left %s, right %s, pixels %s (%s), median_square_width %d, black_border_width %s" %
            #     (self.top, self.bottom, self.left, self.right, pixels, pixels_desc, self.median_square_width,
            #      self.black_border_width))

        assert self.black_border_width < self.median_square_width, \
            "black_border_width %s, median_square_width %s" % (self.black_border_width, self.median_square_width)

    def sanity_check_results(self, contours, debug=False):

        # Verify we found the correct number of squares
        num_squares = len(contours)
        needed_squares = self.size * self.size

        if num_squares < needed_squares:
            # log.debug("sanity False: num_squares %d < needed_squares %d" % (num_squares, needed_squares))
            return False

        elif num_squares > needed_squares:
            # This scenario will need some work so exit here so we notice it
            # log.warning("sanity False: num_squares %d > needed_squares %d" % (num_squares, needed_squares))
            # sys.exit(1)
            return False

        # Verify each row/col has the same number of neighbors
        req_neighbors = self.size - 1

        for con in contours:
            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) = \
                self.get_contour_neighbors(contours, con)

            if row_neighbors != req_neighbors:
                # log.debug("%s sanity False: row_neighbors %d != req_neighbors %s" % (con, row_neighbors, req_neighbors))
                return False

            if col_neighbors != req_neighbors:
                # log.debug("%s sanity False: col_neighbors %d != req_neighbors %s" % (con, col_neighbors, req_neighbors))
                return False

        # log.debug("%s sanity True" % con)
        return True

    def get_mean_row_col_for_index(self, col_index, row_index):
        global mean_Y
        total_X = 0
        total_prev_X = 0
        total_next_X = 0

        total_Y = 0
        total_prev_Y = 0
        total_next_Y = 0

        candidates_X = 0
        candidates_prev_X = 0
        candidates_next_X = 0

        candidates_Y = 0
        candidates_prev_Y = 0
        candidates_next_Y = 0

        for con in self.candidates:
            if con.col_index == col_index - 1:
                total_prev_X += con.cX
                candidates_prev_X += 1

            elif con.col_index == col_index:
                total_X += con.cX
                candidates_X += 1

            elif con.col_index == col_index + 1:
                total_next_X += con.cX
                candidates_next_X += 1

            if con.row_index == row_index - 1:
                total_prev_Y += con.cY
                candidates_prev_Y += 1

            elif con.row_index == row_index:
                total_Y += con.cY
                candidates_Y += 1

            elif con.row_index == row_index - 1:
                total_next_Y += con.cY
                candidates_next_Y += 1

        if candidates_X:
            mean_X = int(total_X / candidates_X)

            if self.debug:
                pass
                # log.info("get_mean_row_col_for_index: col_index %d, total_X %d, candidates_X %d, mean_X %d" % (
                # col_index, total_X, candidates_X, mean_X))

        elif candidates_prev_X:
            mean_X = int(total_prev_X / candidates_prev_X) + self.median_square_width + self.black_border_width

            if self.debug:
                pass
                # log.info(
                #     "get_mean_row_col_for_index: col_index %d, total_prev_X %d, candidates_prev_X %d, mean_X %d" % (
                #     col_index, total_prev_X, candidates_prev_X, mean_X))

        elif candidates_next_X:
            mean_X = int(total_next_X / candidates_next_X) - self.median_square_width - self.black_border_width

            if self.debug:
                pass
                # log.info(
                #     "get_mean_row_col_for_index: col_index %d, total_next_X %d, candidates_next_X %d, mean_X %d" % (
                #     col_index, total_next_X, candidates_next_X, mean_X))

        else:
            if not candidates_X:
                # raise ZeroCandidates("candidates_X is %s, it cannot be zero" % candidates_X)
                raise Exception("candidates_X is %s, it cannot be zero" % candidates_X)

        if candidates_Y:
            mean_Y = int(total_Y / candidates_Y)

            if self.debug:
                pass
                # log.info("get_mean_row_col_for_index: col_index %d, total_Y %d, candidates_Y %d, mean_Y %d" % (
                # col_index, total_Y, candidates_Y, mean_Y))

        elif candidates_prev_Y:
            mean_Y = int(total_prev_Y / candidates_prev_Y) + self.median_square_width + self.black_border_width

            if self.debug:
                pass
                # log.info(
                #     "get_mean_row_col_for_index: col_index %d, total_prev_Y %d, candidates_prev_Y %d, mean_Y %d" % (
                #     col_index, total_prev_Y, candidates_prev_Y, mean_Y))

        elif candidates_next_Y:
            mean_Y = int(total_next_Y / candidates_next_Y) - self.median_square_width - self.black_border_width

            if self.debug:
                pass
                # log.info(
                #     "get_mean_row_col_for_index: col_index %d, total_next_Y %d, candidates_next_Y %d, mean_Y %d" % (
                #     col_index, total_next_Y, candidates_next_Y, mean_Y))

        else:
            if not candidates_Y:
                # raise ZeroCandidates("candidates_Y is %s, it cannot be zero" % candidates_Y)
                raise Exception("candidates_Y is %s, it cannot be zero" % candidates_Y)

        return mean_X, mean_Y

    def find_missing_squares(self, missing_count):

        if self.debug:
            pass
            # log.info("find_missing_squares: size %d, missing %d squares" % (self.size, missing_count))

        needed = []
        con_by_row_col_index = {}

        # ID which row and col each candidate contour belongs to
        for con in self.candidates:
            self.set_contour_row_col_index(con)

            if (con.col_index, con.row_index) in con_by_row_col_index:
                # raise FoundMulitpleContours("(%d, %d) value %s, con %s" % (
                raise Exception("(%d, %d) value %s, con %s" % (
                    con.col_index, con.row_index, con_by_row_col_index[(con.col_index, con.row_index)], con))
            else:
                con_by_row_col_index[(con.col_index, con.row_index)] = con

        if self.debug:
            pass
            # log.info("find_missing_squares: con_by_row_col_index\n%s" % pformat(con_by_row_col_index))

        # Build a list of the row/col indexes where we need a contour
        for col_index in range(self.size):
            for row_index in range(self.size):
                if (col_index, row_index) not in con_by_row_col_index:
                    needed.append((col_index, row_index))

        if self.debug:
            pass
            # log.info("find_missing_squares: missing_count %d, needed %s" % (missing_count, pformat(needed)))

        if len(needed) != missing_count:
            raise Exception(
                "missing_count is %d but needed has %d:\n%s" % (missing_count, len(needed), pformat(needed)))

        # For each of 'needed', create a CustomContour() object at the coordinates
        # where we need a contour. This will be a very small square contour but that
        # is all we need.
        index = len(self.contours_by_index.keys()) + 1
        missing = []

        for (col_index, row_index) in needed:
            (missing_X, missing_Y) = self.get_mean_row_col_for_index(col_index, row_index)

            if self.debug:
                pass
                # log.info("find_missing_squares: contour (%d, %d), coordinates(%d, %d)\n" % (
                #     col_index, row_index, missing_X, missing_Y))

            missing_size = 5
            missing_left = missing_X - missing_size
            missing_right = missing_X + missing_size
            missing_top = missing_Y - missing_size
            missing_bottom = missing_Y + missing_size

            missing_con = np.array([[[missing_left, missing_top],
                                     [missing_right, missing_top],
                                     [missing_right, missing_bottom],
                                     [missing_left, missing_bottom]
                                     ]], dtype=np.int32)

            # log.info("missing_con:\n%s\n" % pformat(missing_con))
            con = CustomContour(self, index, missing_con, None, self.debug)
            missing.append(con)
            index += 1

        return missing

    def remove_outside_contours(self, extra_count):
        contours_to_remove = []

        # TODO this logic needs some more work but I want more test-data entries first
        for con in self.candidates:
            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) = \
                self.get_contour_neighbors(self.candidates, con)
            # log.info("%s row_square_neighbors %s, col_square_neighbors %s" %
            #    (con, row_square_neighbors, col_square_neighbors))

            if not row_square_neighbors or not col_square_neighbors:
                contours_to_remove.append(con)

        for con in contours_to_remove:
            self.candidates.remove(con)

        assert len(contours_to_remove) == extra_count, "%d contours_to_remove but extra_count is %d" % (
            len(contours_to_remove), extra_count)

    def sort_by_row_col(self, contours, size):
        """
        Given a list of contours sort them starting from the upper left corner
        and ending at the bottom right corner
        """
        result = []
        num_squares = len(contours)

        for row_index in range(size):

            # We want the 'size' squares that are closest to the top
            tmp = []
            for con in contours:
                tmp.append((con.cY, con.cX))
            top_row = sorted(tmp)[:size]

            # Now that we have those, sort them from left to right
            top_row_left_right = []
            for (cY, cX) in top_row:
                top_row_left_right.append((cX, cY))
            top_row_left_right = sorted(top_row_left_right)

            # log.debug("sort_by_row_col() row %d: %s" % (row_index, pformat(top_row_left_right)))
            contours_to_remove = []
            for (target_cX, target_cY) in top_row_left_right:
                for con in contours:

                    if con in contours_to_remove:
                        continue

                    if con.cX == target_cX and con.cY == target_cY:
                        result.append(con)
                        contours_to_remove.append(con)
                        break

            for con in contours_to_remove:
                contours.remove(con)

        # assert len(result) == num_squares, "Returning %d squares, it should be %d" % (len(result), num_squares)
        return result

    def analyze(self, image):
        # Размеры изображения
        self.height, self.width = image.shape[:2]
        img_area = int(self.height * self.width)
        # Переводи в грейскейл
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.display(gray, "gray")

        # Сглаживаем
        # no_noise = cv2.fastNlMeansDenoising(gray, 15, 15, 7, 21) Супер, но медленно
        # no_noise = gray.copy() Просто тупо
        no_noise = cv2.blur(gray, (9, 9))
        # kernel = np.ones((15, 15), np.float32)/225
        # return cv2.filter2D(gray, -1, kernel)
        self.display(no_noise, "removed noise")

        # Находим границы
        canny = cv2.Canny(no_noise, 5, 20)
        self.display(canny, "canny")

        # Делаем границы толще
        kernel = np.ones((4, 4), np.uint8)
        dilated = cv2.dilate(canny, kernel, iterations=4)
        self.display(dilated, "dilated")

        (contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            raise Exception("hierarchy")

        # Т.к. hierarchy = [ [ [],[],[],[] ] ].  Почему-то.
        hierarchy = hierarchy[0]

        # zip вернет список пар где на первом месте будет контур, а на втором его иерархия
        for index, component in enumerate(zip(contours, hierarchy)):
            con = CustomContour(self, index, component[0], component[1], self.debug)
            self.contours_by_index[index] = con

            # Можно и больше 30. На средней фотке площадь квадратов около 1300
            # if 3000 > con.area > 30 and con.cX is not None:
            if con.area > 30 and con.cX is not None:
                self.candidates.append(con)
        self.display(image, "count")

        if self.remove_non_square_candidates():
            self.display(image, "50 post non-squares removal #1")

        # Тут начинается страх
        # If a contour is more than 25% of the entire image throw it away
        if self.remove_gigantic_candidates(int(img_area / 4)):
            self.display(image, "60 remove gigantic squares")

        # Find the median square size, we need that in order to find the
        # squares that make up the boundry of the cube
        self.get_median_square_area()

        # Remove contours less than 1/2 the median square size
        if self.remove_dwarf_candidates(int(self.mean_square_area / 2)):
            self.get_median_square_area()
            self.display(image, "70 remove dwarf squares")

        # Sometimes we find a square within a square due to the black space
        # between the cube squares.  Throw away the outside square (it contains
        # the black edge) and keep the inside square.
        if self.remove_square_within_square_candidates():
            self.get_median_square_area()
            self.display(image, "80 post square-within-square removal #1")

        # Remove contours more than 2x the mean square size
        if self.remove_gigantic_candidates(int(self.mean_square_area * 2)):
            self.get_median_square_area()
            self.display(image, "90 remove larger squares")

        if not self.get_cube_boundary(False):
            raise Exception("%s could not find the cube boundry" % self)

        # remove all contours that are outside the boundary of the cube
        if self.remove_contours_outside_cube(self.candidates):
            self.get_median_square_area()
            self.display(self, "100 post outside cube removal")

        # Find the cube size (3x3x3, 4x4x4, etc)
        # if cube_size:
        #     self.size = cube_size
        # else:
        #     if not self.get_cube_size():
        #         raise CubeNotFound("%s invalid cube size %sx%sx%s" % (self.name, self.size, self.size, self.size))
        cube_size = 3
        self.size = cube_size

        # Now that we know the cube size, re-define the boundry
        if not self.get_cube_boundary(True):
            raise Exception("%s could not find the cube boundry" % self)
            # raise CubeNotFound("%s could not find the cube boundry" % self.name)

        # remove contours outside the boundry
        if self.remove_contours_outside_cube(self.candidates):
            self.get_median_square_area()
            self.display(image, "110 post outside cube removal")

        # Now that we know the median size of each square, go back and
        # remove non-squares one more time
        # if self.remove_non_square_candidates(self.median_square_area):
        #    self.display_candidates(self.image, "120 post non-square-candidates removal")

        # We just removed contours outside the boundry and non-square contours, re-define the boundry
        # if not self.get_cube_boundry(True):
        #    raise CubeNotFound("%s could not find the cube boundry" % self.name)

        # Find the size of the gap between two squares
        self.get_black_border_width()
        missing = []

        if not self.sanity_check_results(self.candidates):

            # How many squares are missing?
            missing_count = (self.size * self.size) - len(self.candidates)

            # We are short some contours
            if missing_count > 0:
                missing = self.find_missing_squares(missing_count)

                if missing:
                    self.candidates.extend(missing)
                else:
                    return False
                    # if webcam:
                    #     return False
                    # else:
                    #     log.info("Could not find missing squares needed to create a valid cube")
                    #     raise Exception("Unable to extract image from %s" % self.name)

            # We have too many contours
            elif missing_count < 0:
                self.remove_outside_contours(missing_count * -1)

            else:
                raise Exception("missing_count %s we should not be here" % missing_count)

        self.display(image, "150 Final", missing)

        raw_data = []
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for con in self.sort_by_row_col(deepcopy(self.candidates), self.size):
            # Use the mean value of the contour
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [con.contour], 0, 255, -1)
            # (mean_blue, mean_green, mean_red, _) = map(int, cv2.mean(image, mask=mask))
            (mean_h, mean_s, mean_v, _) = map(int, cv2.mean(hsv_image, mask=mask))
            # raw_data.append((mean_red, mean_green, mean_blue))
            raw_data.append((mean_h, mean_s, mean_v))

        if self.debug:
            pass
            # log.info("squares RGB data\n%s\n" % pformat(raw_data))

        squares_per_side = len(raw_data)
        size = int(math.sqrt(squares_per_side))
        # init_square_index = (self.index * squares_per_side) + 1
        init_square_index = (0 * squares_per_side) + 1

        square_indexes = []
        for row in range(size):
            square_indexes_for_row = []
            for col in range(size):
                square_indexes_for_row.append(init_square_index + (row * size) + col)
            square_indexes.append(square_indexes_for_row)

        if self.debug:
            # log.info("%s square_indexes\n%s\n" % (self, pformat(square_indexes)))
            square_indexes = compress_2d_array(square_indexes)
            # log.info("%s square_indexes (final)\n%s\n" % (self, pformat(square_indexes)))
        else:
            square_indexes = compress_2d_array(square_indexes)

        self.data = {}
        for index in range(squares_per_side):
            square_index = square_indexes[index]
            # (red, green, blue) = raw_data[index]
            (h, s, v) = raw_data[index]

            if self.debug:
                pass
                # log.info("square %d RGB (%d, %d, %d)" % (square_index, red, green, blue))
            # print("square %d RGB (%d, %d, %d)" % (square_index, red, green, blue))

            # self.data is a dict where the square number (as an int) will be
            # the key and a RGB tuple the value
            # self.data[square_index] = (red, green, blue)
            self.data[square_index] = (h, s, v)
        # print(self.data)
        # pformat(raw_data)
        # log.debug("")

        # If we made it to here it means we were able to extract the cube from
        # the image with the current gamma setting so no need to look any further
        return True

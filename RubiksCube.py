import math

import cv2
import numpy as np


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


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
        angle_A = int(math.degrees(get_angle(C, B, A)))
        if angle_A < min_angle or angle_A > max_angle:
            return False

        # Angle at B
        angle_B = int(math.degrees(get_angle(A, D, B)))
        if angle_B < min_angle or angle_B > max_angle:
            return False

        # Angle at C
        angle_C = int(math.degrees(get_angle(A, D, C)))
        if angle_C < min_angle or angle_C > max_angle:
            return False

        # Angle at D
        angle_D = int(math.degrees(get_angle(C, B, D)))
        if angle_D < min_angle or angle_D > max_angle:
            return False

        far_left = min(A[0], B[0], C[0], D[0])
        far_right = max(A[0], B[0], C[0], D[0])
        far_up = min(A[1], B[1], C[1], D[1])
        far_down = max(A[1], B[1], C[1], D[1])
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
        if B[1] < A[1]:
            # Angle at B relative to the AB line
            angle_B = int(math.degrees(get_angle(A, top_left, B)))

            if debug:
                log.info("AB is horizontal, angle_B %s, ROTATE_THRESHOLD %s" % (angle_B, rotate_threshold))

            if angle_B > rotate_threshold:
                if debug:
                    log.info("AB horizontal rotation %s is above ROTATE_THRESHOLD %s" % (angle_B, rotate_threshold))
                return False
        else:
            # Angle at A relative to the AB line
            angle_A = int(math.degrees(get_angle(B, top_right, A)))

            if debug:
                log.info("AB is vertical, angle_A %s, ROTATE_THRESHOLD %s" % (angle_A, rotate_threshold))

            if angle_A > rotate_threshold:
                if debug:
                    log.info("AB vertical rotation %s is above ROTATE_THRESHOLD %s" % (angle_A, rotate_threshold))
                return False

        # TODO - if the area of the approx is way more than the
        # area of the contour then this is not a square
        return True

    def __str__(self):
        return "Contour #%d (%s, %s)" % (self.index, self.cX, self.cY)


class RubiksCube:
    def __init__(self, debug=False):
        self.debug = debug
        self.candidates = []
        self.width = None
        self.height = None
        self.raw_contours = None
        self.hierarchy = None
        self.contours_by_index = {}

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

                # if con.is_square():
                #     to_draw_square.append(con.contour)
                #     to_draw_approx.append(con.approx)
                # else:
                to_draw.append(con.contour)
                to_draw_approx.append(con.approx)

            tmp_image = image.copy()
            # cons that are squares are in green
            # for non-squqres the approx is green and contour is blue
            cv2.drawContours(tmp_image, to_draw, -1, (255, 0, 0), 2)
            cv2.drawContours(tmp_image, to_draw_approx, -1, (0, 0, 255), 2)
            # cv2.drawContours(tmp_image, to_draw_square, -1, (0, 255, 0), 2)

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

    def analyze(self, image):
        # Размеры изображения
        self.height, self.width = image.shape[:2]

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
        ap = [c.approx for c in self.candidates]
        image = cv2.drawContours(image, ap, -1, (255, 0, 0))
        self.candidates.clear()
        return image
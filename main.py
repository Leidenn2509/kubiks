import numpy as np
import cv2
from math import sqrt
from matplotlib import pyplot as plt


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


# static double angle( Point pt1, Point pt2, Point pt0 )
# {
#     double dx1 = pt1.x - pt0.x;
#     double dy1 = pt1.y - pt0.y;
#     double dx2 = pt2.x - pt0.x;
#     double dy2 = pt2.y - pt0.y;
#     return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
# }
def angle(p1, p2, p0):
    dx1 = p1[0] - p0[0]
    dy1 = p1[1] - p0[1]
    dx2 = p2[0] - p0[0]
    dy2 = p2[1] - p0[1]
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for grayt in cv2.split(img):
        for tt in xrange(0, 255, 26):
            if tt == 0:
                b = cv2.Canny(grayt, 0, 50, apertureSize=5)
                b = cv2.dilate(b, None)
            else:
                _retval, b = cv2.threshold(grayt, tt, 255, cv2.THRESH_BINARY)
            cou, _hierarchy = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cntt in cou:
                cnt_lent = cv2.arcLength(cntt, True)
                cntt = cv2.approxPolyDP(cntt, 0.02 * cnt_lent, True)
                if len(cntt) == 4 and cv2.contourArea(cntt) > 1000 and cv2.isContourConvex(cntt):
                    cntt = cntt.reshape(-1, 2)
                    max_cost = np.max([angle_cos(cntt[ii], cntt[(ii + 1) % 4], cntt[(ii + 2) % 4]) for ii in xrange(4)])
                    if max_cost < 0.1:
                        squares.append(cntt)
    return squares


# def draw_rect(frame, squares):
#     for square in squares:


def find(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5, 1.5)
    # gray = cv2.Canny(gray, 0, 30, 3)
    gray = cv2.Canny(gray, 50, 200, 3)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sq = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 9, True)
        sq.append(approx)
        size = approx.size
        area = abs(cv2.contourArea(approx))
        convex = cv2.isContourConvex(approx)
        if size == 4:
            print "area " + str(area)
            sq.append(approx)
            if area > 5:
                print "\tsize " + str(size) + "; area " + str(area)
                if convex:
                    print "\t\tsize " + str(size) + "; area " + str(area) + "; convex " + str(convex)
        # if approx.size == 4 and abs(cv2.contourArea(approx)) > 5 and cv2.isContourConvex(approx):
        #     maxCosine = 0
        #     for i in range(2, 5):
        #         cosine = abs(angle(approx[i % 4], approx[i - 2], approx[i - 1]))
        #         maxCosine = max(maxCosine, cosine)
        #     if maxCosine < 0.3:
        #         sq.append(approx)
    return sq


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # ******************************************************************* do something
        sq = find(frame)
        cv2.drawContours(frame, sq, -1, (0, 255, 0), 3)

        # *******************************************************************
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    #
    #
    #
    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     if not ret:
    #         continue
    #     # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     blur = cv2.GaussianBlur(gray, (7, 7), 0)
    #     ret3, f = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    #     # f = cv2.Canny(gray, 20, 200)
    #     contours, hierarchy = cv2.findContours(f, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     sq = []
    #     for cnt in contours:
    #         cnt_len = cv2.arcLength(cnt, True)
    #         cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
    #         if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
    #             cnt = cnt.reshape(-1, 2)
    #             max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in xrange(4)])
    #             if max_cos < 0.1:
    #                 sq.append(cnt)
    #     cv2.drawContours(frame, sq, -1, (0, 255, 0), 3)
    #     # Display the resulting frame
    #     cv2.imshow('frame', frame)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break

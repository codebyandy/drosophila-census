
import cv2 as cv
import numpy as np
import sys
import os
import timeit


def processDir(read_dir, write_dir):
    counts_file = os.path.join(write_dir, 'counts.csv')
    
    if os.path.exists(counts_file):
        val = ""
        while val != "y" and val != "n":
            val = input("Look's like a counts.csv file already exists for this directory. Are you sure you want to proceed? [y/n] ")
        if val == "n": return
    
    f = open(counts_file, "w")
    f.write("image,count,threshold_min,size_min,size_max,flag,time\n")
    f.close()

    images = []
    for filename in os.listdir(read_dir):
        if os.path.isfile(os.path.join(read_dir, filename)):
            images.append(filename)
    
    for image in images: processImage(image, read_dir, write_dir)


def update(x): pass


def processImage(filename, read_dir, write_dir):
    start = timeit.default_timer()
    print("Processing ", filename, " in ", read_dir)

    cv.namedWindow("Trackbars")
    cv.namedWindow("Image processing (" + filename + ")", cv.WINDOW_NORMAL)
    cv.createTrackbar('ThreshMin', 'Trackbars', 90, 255, update)
    cv.createTrackbar('SizeMin', 'Trackbars', 10, 40, update)
    cv.createTrackbar('SizeMax', 'Trackbars', 100, 400, update)
    cv.createTrackbar('Flag', 'Trackbars', 0, 1, update)

    while True:
        path = os.path.join(read_dir, filename)
        img = cv.imread(path)

        gray_crop = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh_crop = cv.threshold(gray_crop, 0, 255, cv.THRESH_OTSU)
        bbox = cv.boundingRect(thresh_crop)
        x, y, w, h = bbox
        foreground = img[y:y+h, x:x+w]
        img = foreground.copy()
        
        tmin = cv.getTrackbarPos('ThreshMin', 'Trackbars')
        smin = cv.getTrackbarPos('SizeMin', 'Trackbars')
        smax = cv.getTrackbarPos('SizeMax', 'Trackbars')
        flag = cv.getTrackbarPos('Flag', 'Trackbars')
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, tmin, 255, cv.THRESH_BINARY_INV)
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, 4, cv.CV_32S)

        arr = stats[:, cv.CC_STAT_AREA]
        selected = np.logical_and(arr > smin, arr < smax)
        count = np.count_nonzero(selected)

        bboxes = []
        for i in range(0, num_labels):
            if selected[i] == 0: continue
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            area = stats[i, cv.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv.circle(img, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            bboxes.append(' '.join([str(x), str(y), str(w), str(h)]))

        thresh = np.stack((thresh,) * 3, axis=-1)
        images = [thresh, img]
        win_names = ['Threshold', 'Selections'] 
        font = cv.FONT_HERSHEY_SIMPLEX
        img_stack = np.hstack(images)
        for index, name in enumerate(win_names):
            image = cv.putText(img_stack, f'{index + 1}. {name}', (5 + img.shape[1] * index, 30),
                                font, 1, (205, 0, 255), 2, cv.LINE_AA)

        cv.imshow("Image processing (" + filename + ")", img_stack)
        cv.resize(img_stack, (1920, 1080))
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            stop = timeit.default_timer()
            img_name = filename[:filename.rfind('.')]

            cv.imwrite(os.path.join(write_dir, img_name + '_thresh.png'), thresh)
            cv.imwrite(os.path.join(write_dir, img_name + '_selection.png'), img)
            
            combo = np.concatenate((foreground, thresh, img), axis=1)
            cv.imwrite(os.path.join(write_dir, img_name + '_combo.png'), combo)

            f = open(os.path.join(write_dir, img_name + '_bboxes.txt'), "w")
            for i, bbox in enumerate(bboxes):
                f.write(str(i) + " " + bbox + "\n")
            f.close()

            f = open(os.path.join(write_dir, 'counts.csv'), "a")
            f.write(','.join([img_name, str(count), str(tmin), str(smin), str(smax), str(flag), str(stop - start)]) + "\n")
            f.close()

            break

    cv.destroyAllWindows()


def main():
    args = sys.argv
    read_dir, write_dir = args[2], args[3]

    if args[1] == "-dir":
        processDir(read_dir, write_dir)
    if args[1] == "-file":
        file=  args[4]
        processImage(file, read_dir, write_dir)

if __name__ == "__main__":
    main()
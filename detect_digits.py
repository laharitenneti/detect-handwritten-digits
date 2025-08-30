import cv2
import numpy as np

'''TRAINING'''
#load training image
img = cv2.imread('images/digits.png')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
downsize = cv2.pyrDown(img)

# split training image into 5000 calls: 50 columns x 100 rows
cells = [np.hsplit(row, 100) for row in np.vsplit(grey, 50)]
x = np.array(cells)

# creating training & testing datasets
train = x[:, :70].reshape(-1, 400).astype(np.float32)   # (3500,400)
test = x[:, 70:100].reshape(-1, 400).astype(np.float32) # (1500,400)

# labels
k = np.arange(10)
train_labels = np.repeat(k, 350)[:, np.newaxis]
test_labels = np.repeat(k, 150)[:, np.newaxis]

# Train KNN
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# Evaluate accuracy
ret, result, neighbours, dist = knn.findNearest(test, k=3)
matches = result == test_labels
accuracy = np.count_nonzero(matches) * 100.0 / result.size
print(f"Training dataset accuracy = {accuracy:.2f}%")

'''PREPARING INPUT IMAGE'''
#for sorting contours from left to right
def xcoord_contour(contour):
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        return (int(M['m10']/M['m00']))

#for making a square out of the given image
def sq(not_sq):
    black = [0,0,0]
    dims = not_sq.shape
    h = dims[0]
    w = dims[1]
    if (h == w):
        square = not_sq
        return square
    else:
        doublesize = cv2.resize(not_sq,(2*w, 2*h), interpolation = cv2.INTER_CUBIC)
        h *= 2
        w *= 2
        if (h > w):
            pad = int((h - w)/2)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=black)
        else:
            pad = int((w - h)/2)
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value=black)
    return doublesize_square

#resizing image to 20x20 pixels
def resize_to_pixel(dims, img):
    buffer = 4
    dims  -= buffer
    squared = img
    r = float(dims) / squared.shape[1]
    dim = (dims, int(squared.shape[0] * r))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    dims2 = resized.shape
    h_r = dims2[0]
    w_r = dims2[1]
    black = [0,0,0]
    if (h_r > w_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=black)
    if (h_r < w_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=black)
    p = 2
    resized = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=black)
    return resized

if __name__ == "__main__":

    img = cv2.imread('images/num_test.jpeg')
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = xcoord_contour, reverse = False)

    full_number = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)    

        if w >= 5 and h >= 25:
            roi = blurred[y:y + h, x:x + w]
            ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)

            squared = sq(roi)
            final = resize_to_pixel(20, squared)
            cv2.imshow("final", final)
            final_array = final.reshape((1,400))
            final_array = final_array.astype(np.float32)
            ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
            
            number = str(int(float(result[0])))
            full_number.append(number)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, number, (x , y + 155), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
            cv2.imshow("image", img)
            cv2.waitKey(0) 
            
    cv2.destroyAllWindows()
    print ("The number is: " + ''.join(full_number))

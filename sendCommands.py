from ultralytics import YOLO
from bitarray import bitarray
import time

model = YOLO("Model/best.pt")

#x1y1 va checker gauche/dessous
#x2y2 va checker droite/haut

#Gauche et/ou en dessous

results = model("Images/1.jpg", agnostic_nms=True)

start_time = time.time()

def isConfidenceHigher(x, currentBestBox):
    return x[4] > currentBestBox[4]

currentBestBox = None
for i in results:
    x = i.boxes[0].boxes.numpy()[0]
    #x = [float(y) for y in x]
    if currentBestBox == None or isConfidenceHigher(x, currentBestBox):
        currentBestBox = x
    #       *           (0,1)
    #
    #            *      (2,3)
    # image size en xy  (4,5)
a = 4 * bitarray('0')
b = 4 * bitarray('0')


# 215 - 128 - 64 sont nos limites, placeholder values
def isValueUnderThreshold(value, byte):
    if value < 215:
        byte[1] = 1
    elif value < 128:
        byte[2] = 1
    elif value < 64:
        byte[3] = 1
    else:
        return False
    return True

#Droite et/ou en haut
#Todo: screenSize -> 640 vu que c'est toujours 640p?
def isValueOverThreshold(value, screenSize, byte):
    if value > screenSize - 215:
        byte[1] = 1
    elif value > screenSize - 128:
        byte[2] = 1
    elif value > screenSize - 64:
        byte[3] = 1
    else:
        return False
    return True

def computeBoxArea(height, width):
    return height * width


if not isValueUnderThreshold(currentBestBox[0], a):
    if isValueOverThreshold(currentBestBox[2], 640, a):
        a[0] = 1
if not isValueUnderThreshold(currentBestBox[3], b):
    if isValueOverThreshold(currentBestBox[1], 640, b):
        b[0] = 1
a.extend(b)
print(a)
print("--- %s seconds ---" % (time.time() - start_time))
import cv2
import time
import imutils
import numpy as np
def style_transfer(weights, image):
    net = cv2.dnn.readNetFromTorch(weights)
    #frame = cv2.imread(image)
    current_image = image.copy()
    current_image = imutils.resize(current_image, 224)
    blob = cv2.dnn.blobFromImage(current_image, 1.0, (224, 224), (103.939, 116.779, 123.680), swapRB=False,crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end_time = time.time()
    # print(f'{1/(end_time-start)} FPS')
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680

    # print(f'before output shape {output.shape}')
    # torch.tanspose(目标坐标顺序)
    # output = output.transpose(1, 2, 0).astype(np.uint8)
    output = output.transpose(1, 2, 0)
    # output = output.astype(np.uint8)
    # print(f'before output shape {output.shape}')
    
    #style = weights.split('/')[-1].split('.')[0]
    #savename = image.split('/')[-1].split('.')[0] + '_' + style + '.jpg'
    #cv2.imwrite(savename, output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    return output

if __name__ == '__main__':
    # weights_path = 'models/udnie.t7'
    weights_path = 'models/the_wave.t7'
    image = 'test.jpg'
    style_transfer(weights_path, image)
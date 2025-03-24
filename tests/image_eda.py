import os, sys
sys.path.append(os.getcwd())

import cv2

def main():
    images_path = 'data/welding/train/images'
    labels_path = 'data/welding/train/labels'
    
    '''
    num = 10
    print(os.listdir(images_path)[num])
    
    image_path = os.path.join(images_path, os.listdir(images_path)[num])
    
    label_path = os.path.join(labels_path, os.listdir(labels_path)[num])
    '''
    # image_path = 'data/welding_images/train/images/1_jpg.rf.26123444e7a0e4f9f903bbc30f62f2be.jpg'
    # label_path = 'data/welding_images/train/labels/1_jpg.rf.26123444e7a0e4f9f903bbc30f62f2be.txt'
    
    image_path = 'data/welding_images/train/images/22_jpg.rf.1d5b9e4fce3db69b67b8e27f8346f3ee.jpg'
    label_path = 'data/welding_images/train/labels/22_jpg.rf.1d5b9e4fce3db69b67b8e27f8346f3ee.txt'
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, channel = img.shape

    with open(label_path, 'r') as f:
        label = f.read().split()

    coord_li = []
    for x, y in zip(label[1::2], label[2::2]):
        x, y = float(x), float(y)
        x_, y_ = int(width * x), int(height * y)
        coord_li.append((x_, y_))
    
    coord_li = sorted(coord_li, key=lambda x: x[0])
        
    for i in range(0, len(coord_li) - 1):
        p1, p2 = coord_li[i], coord_li[i+1]
        cv2.line(img, p1, p2, color=(0, 0, 255), thickness=2)
        
    cv2.imwrite('image_plot_test.jpg', img)
    
    
if __name__ == '__main__':
    main()
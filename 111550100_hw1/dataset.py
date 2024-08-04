import os
import cv2
import glob
import numpy as np



def load_data_small():
    """
        This function loads images form the path: 'data/data_small' and return the training
        and testing dataset. The dataset is a list of tuples where the first element is the 
        numpy array of shape (m, n) representing the image the second element is its 
        classification (1 or 0).

        Parameters:
            None

        Returns:
            dataset: The first and second element represents the training and testing dataset respectively
    """

    # Begin your code (Part 1-1)
    """
    1. Initialize the empty list (tarining_dataset, testing_dataset).
    2. Use os.listdir() and for loop to traverse all the image.
    3. Use os.path.join() to generate the file path of each image file.
    4. Use cv2.imread() to read the image, and change it to gray scale image.
    5. Append the image and its classification into dataset.
    6. Since there are floders(face, non-face) in train and test, so do the above 4 times.
    7. Return the dataset.
    """
    tarining_dataset = []
    testing_dataset = []

    for file in os.listdir('data/data_small/train/face'):
        image_path = os.path.join('data/data_small/train/face', file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_array = np.array(image)
            tarining_dataset.append((image_array, 1))

    for file in os.listdir('data/data_small/train/non-face'):
        image_path = os.path.join('data/data_small/train/non-face', file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_array = np.array(image)
            tarining_dataset.append((image_array, 0))
    
    for file in os.listdir('data/data_small/test/face'):
        image_path = os.path.join('data/data_small/test/face', file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_array = np.array(image)
            testing_dataset.append((image_array, 1))

    for file in os.listdir('data/data_small/test/non-face'):
        image_path = os.path.join('data/data_small/test/non-face', file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_array = np.array(image)
            testing_dataset.append((image_array, 0))
            
    dataset = (tarining_dataset, testing_dataset)

    # End your code (Part 1-1)
    
    return dataset


def load_data_FDDB(data_idx="01"):
    """
        This function generates the training and testing dataset  form the path: 'data/data_small'.
        The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
        representing the image the second element is its classification (1 or 0).
        
        In the following, there are 4 main steps:
        1. Read the .txt file
        2. Crop the faces using the ground truth label in the .txt file
        3. Random crop the non-faces region
        4. Split the dataset into training dataset and testing dataset
        
        Parameters:
            data_idx: the data index string of the .txt file

        Returns:
            train_dataset: the training dataset
            test_dataset: the testing dataset
    """

    with open("data/data_FDDB/FDDB-folds/FDDB-fold-{}-ellipseList.txt".format(data_idx)) as file:
        line_list = [line.rstrip() for line in file]

    # Set random seed for reproducing same image croping results
    np.random.seed(0)

    face_dataset, nonface_dataset = [], []
    line_idx = 0

    # Iterate through the .txt file
    # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
    while line_idx < len(line_list):
        img_gray = cv2.imread(os.path.join("data/data_FDDB", line_list[line_idx] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        num_faces = int(line_list[line_idx + 1])

        # Crop face region using the ground truth label
        face_box_list = []
        for i in range(num_faces):
            # Here, each face is denoted by:
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
            coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]
            x, y = coord[3] - coord[1], coord[4] - coord[0]            
            w, h = 2 * coord[1], 2 * coord[0]

            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            face_box_list.append([left_top, right_bottom])
            # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

        line_idx += num_faces + 2

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have alreadly save the bounding box of faces into `face_box_list`, you can utilize it for non-face region cropping
        for i in range(num_faces):
            # Begin your code (Part 1-2)
            """
            1. Use `height` and `width` to be the height and width of the image.
            2. Use while loop to check the nonface_box doesn't fully cover the face part.
            3. Use np.random.randint() to check the region I cut is 19*19.
            4. Use .copy() to copy the image.
            """
            height, width = img_gray.shape
            
            nonface_box = None
            
            while nonface_box is None or any((not(nonface_box[0][0] > face_box[1][0] or nonface_box[1][0] < face_box[0][0] or nonface_box[0][1] > face_box[1][1] or nonface_box[1][1] < face_box[0][1])) for face_box in face_box_list):
                w = np.random.randint(19, width)
                h = np.random.randint(19, height)
                x = np.random.randint(0, width - w)
                y = np.random.randint(0, height - h)

                nonface_box = ((x, y), (x + w, y + h))
            left_top, right_bottom = nonface_box
            nonface_img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            # End your code (Part 1-2)

            nonface_dataset.append((cv2.resize(nonface_img_crop, (19, 19)), 0))

        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)

    # train test split
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = face_dataset[:int(SPLIT_RATIO * num_face_data)] + nonface_dataset[:int(SPLIT_RATIO * num_nonface_data)]
    test_dataset = face_dataset[int(SPLIT_RATIO * num_face_data):] + nonface_dataset[int(SPLIT_RATIO * num_nonface_data):]

    return train_dataset, test_dataset


def create_dataset(data_type):
    if data_type == "small":
        return load_data_small()
    else:
        return load_data_FDDB()

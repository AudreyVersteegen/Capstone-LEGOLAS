import os

import PIL.Image
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import math
import PIL.Image as Image
from torchvision import transforms, datasets, models

num_classes = 4

def load_model(filepath):
  data_transform = transforms.Compose(
        [  # transforms.Compose : a class that calls the functions in a list consecutively
            transforms.ToTensor()  # ToTensor : convert numpy image to torch.Tensor type
        ])
  if os.path.isfile(filepath):
    if filepath.endswith('.pth') or filepath.endswith('pt'):
      model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

      if torch.cuda.is_available():
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        device = torch.device("cuda")
        model.load_state_dict(filepath)
      else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        model.load_state_dict(torch.load(filepath, map_location=('cpu')))

      return model, data_transform
    else:
      print('Not a model.')
  else:
    print('Not a valid filepath.')

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold :
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

def preprocess_image(img):
    im = Image.open(img)
    width, height = im.size

    left = width * 0.35
    right = width - (width * 0.45)
    top = height * 0.4
    bottom = height - (height * 0.3)

    cropped_img = im.crop((left, top, right, bottom))
    return cropped_img

def convert_to_coords(pointList):
    # take in a list of [xmin, ymin, xmax, ymax] and return a list of [x, y] coordinates
    coordinates = []
    xs = []
    ys = []
    for box in pointList:
        if len(box) == 4:
            x = (box[0] + box[2]) / 2
            y = (box[1] + box[3]) / 2
            box_coord = [x, y]
            coordinates.append(box_coord)
            xs.append(x)
            ys.append(y)
        else:
            print("Sublist is not the proper size. Should be len 4")
    return coordinates, xs, ys

def get_neighbors(pointList):
    max_neighbor_distance = 300
    min_neighbot_distance = 200
    neighbors_list = []
    for point in pointList:
        # find their neighbors
        neighbors = []
        for other_point in pointList:
            if other_point != point:
                if math.dist(point, other_point) <= max_neighbor_distance and math.dist(point, other_point) >= min_neighbot_distance:
                    neighbors.append(pointList.index(other_point))
        neighbors_list.append(neighbors)

    return neighbors_list

def get_angle(point1, point2):
    x = point2[0]-point1[0]
    y = point2[1]-point1[1]
    angle = math.degrees(math.atan(y/x))
    return angle

def verify_vertical(point1, point2):
    angle = get_angle(point1, point2)
    if abs(angle) > 45:
        return True
    return False

def make_vertical_groups(pointsList, neighborLists):
    vert_groups = []
    in_vert = [False]*len(pointsList)

    for i in range(len(pointsList)):
        point = pointsList[i]
        if not in_vert[i]:
            for neighbor in neighborLists[i]:
                if verify_vertical(point, pointsList[neighbor]):
                    for line in vert_groups:
                        if line.count(pointsList[neighbor]) > 0:
                            line.append(point)
                            in_vert[i] = True
                    if not in_vert[i]:
                        line = [point, pointsList[neighbor]]
                        in_vert[i] = True
                        in_vert[neighbor] = True
                        vert_groups.append(line)
    return vert_groups

def make_horiz_groups(pointsList, neighborLists):
    horiz_groups = []
    in_horiz = [False]*len(pointsList)

    for i in range(len(pointsList)):
        point = pointsList[i]
        if not in_horiz[i]:
            for neighbor in neighborLists[i]:
                if not verify_vertical(point, pointsList[neighbor]):
                    for line in horiz_groups:
                        if line.count(pointsList[neighbor]) > 0:
                            line.append(point)
                            in_horiz[i] = True
                    if not in_horiz[i]:
                        line = [point, pointsList[neighbor]]
                        in_horiz[i] = True
                        in_horiz[neighbor] = True
                        horiz_groups.append(line)
    return horiz_groups

def calculate_best_fit(pointGroups):
    lines = []
    for line in pointGroups:
        x = []
        y = []
        for point in line:
            x.append(point[0])
            y.append(point[1])

        a, b = np.polyfit(x, y, 1)

        equation = [a, b]
        lines.append(equation)
    return lines

def make_grid_lines(pointList):
    # grace of 155 between points
    neighbors = get_neighbors(pointList)

    horiz_groups = make_horiz_groups(pointList, neighbors)
    vert_groups = make_vertical_groups(pointList, neighbors)

    vert_lines = calculate_best_fit(vert_groups)
    horiz_lines = calculate_best_fit(horiz_groups)

    return vert_groups, horiz_groups, vert_lines, horiz_lines

def plot_grid(x_coords, y_coords, vert_lines, horiz_lines):
    plt.plot(x_coords, y_coords, 'o')
    plt.ylim(0, 928)
    plt.xlim(0, 900)

    for line in vert_lines:
        a = line[0]
        b = line[1]
        x = np.linspace(0, 936)
        y = a * x + b
        plt.plot(x, y)

    for line in horiz_lines:
        a = line[0]
        b = line[1]
        x = np.linspace(0, 936)
        y = a * x + b
        plt.plot(x, y)

    plt.imshow(processed_image)
    plt.plot(x_coords, y_coords, 'o')
    plt.ylim(0, 777)
    plt.xlim(0, 936)
    plt.show()

def get_distance(point, vert_lines, horiz_lines):
    


# Autonomous Driving - Car Detection

In this project, I will implement object detection using the very powerful YOLO model. Many of the ideas in this notebook are described in the two YOLO papers: [Redmon et al., 2016](https://arxiv.org/abs/1506.02640) and [Redmon and Farhadi, 2016](https://arxiv.org/abs/1612.08242). 

**Project Objectives**:

- Detect objects in a car detection dataset
- Implement non-max suppression to increase accuracy
- Implement intersection over union
- Handle bounding boxes, a type of image annotation popular in deep learning

## Table of Contents

- [Packages](#0)
- [1 - Problem Statement](#1)
- [2 - YOLO](#2)
    - [2.1 - Model Details](#2-1)
    - [2.2 - Filtering with a Threshold on Class Scores](#2-2)
        - [yolo_filter_boxes](#-)
    - [2.3 - Non-max Suppression](#2-3)
        - [iou](#-)
    - [2.4 - YOLO Non-max Suppression](#2-4)
        - [yolo_non_max_suppression](#-)
    - [2.5 - Wrapping Up the Filtering](#2-5)
        - [yolo_eval](#-)
- [3 - Test YOLO Pre-trained Model on Images](#3)
    - [3.1 - Defining Classes, Anchors and Image Shape](#3-1)
    - [3.2 - Loading a Pre-trained Model](#3-2)
    - [3.3 - Convert Output of the Model to Usable Bounding Box Tensors](#3-3)
    - [3.4 - Filtering Boxes](#3-4)
    - [3.5 - Run the YOLO on an Image](#3-5)
- [4 - Summary for YOLO](#4)
- [5 - References](#5)

<a name='0'></a>
## Packages


```python
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

%matplotlib inline
```

<a name='1'></a>
## 1 - Problem Statement

You are working on a self-driving car. Go you! As a critical component of this project, you'd like to first build a car detection system. To collect data, you've mounted a camera to the hood of the car, which takes pictures of the road ahead every few seconds as you drive around. 

<center>
<video width="400" height="200" src="nb_images/road_video_compressed2.mp4" type="video/mp4" controls>
</video>
</center>

<caption><center> Pictures taken from a car-mounted camera while driving around Silicon Valley. <br> Dataset provided by <a href="https://www.drive.ai/">drive.ai</a>.
</center></caption>

You've gathered all these images into a folder and labelled them by drawing bounding boxes around every car you found. Here's an example of what your bounding boxes look like:

<img src="nb_images/box_label.png" style="width:500px;height:250;">
<caption><center> <u><b>Figure 1</u></b>: Definition of a box<br> </center></caption>

If there are 80 classes you want the object detector to recognize, you can represent the class label $c$ either as an integer from 1 to 80, or as an 80-dimensional vector (with 80 numbers) one component of which is 1, and the rest of which are 0. 

YOLO ("You Only Look Once") performs object detection, and then apply it to car detection. Because the YOLO model is very computationally expensive to train, the pre-trained weights are already loaded for use. 

<a name='2'></a>
## 2 - YOLO

"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

<a name='2-1'></a>
### 2.1 - Model Details

#### Inputs and outputs
- The **input** is a batch of images, and each image has the shape (m, 608, 608, 3)
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers $(p_c, b_x, b_y, b_h, b_w, c)$ . If you expand $c$ into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 

#### Anchor Boxes
* Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes.  Use 5 anchor boxes (to cover the 80 classes), stored in the file './model_data/yolo_anchors.txt'
* The dimension of the encoding tensor of the second to last dimension based on the anchor boxes is $(m, n_H,n_W,anchors,classes)$.
* The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).  


#### Encoding
this encoding represents:. 

<img src="nb_images/architecture.png" style="width:700px;height:400;">
<caption><center> <u><b> Figure 2 </u></b>: Encoding architecture for YOLO<br> </center></caption>

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since I am using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, I'll flatten the last two dimensions of the shape (19, 19, 5, 85) encoding, so the output of the Deep CNN becomes (19, 19, 425).

<img src="nb_images/flatten.png" style="width:700px;height:400;">
<caption><center> <u><b> Figure 3 </u></b>: Flattening the last two last dimensions<br> </center></caption>

#### Class score

Now, for each box (of each cell) I'll compute the following element-wise product and extract a probability that the box contains a certain class.  
The class score is $score_{c,i} = p_{c} \times c_{i}$: the probability that there is an object $p_{c}$ times the probability that the object is a certain class $c_{i}$.

<img src="nb_images/probability_extraction.png" style="width:700px;height:400;">
<caption><center> <u><b>Figure 4</u></b>: Find the class detected by each box<br> </center></caption>

##### Example of figure 4
* In figure 4, let's say for box 1 (cell 1), the probability that an object exists is $p_{1}=0.60$.  So there's a 60% chance that an object exists in box 1 (cell 1).  
* The probability that the object is the class "category 3 (a car)" is $c_{3}=0.73$.  
* The score for box 1 and for category "3" is $score_{1,3}=0.60 \times 0.73 = 0.44$.  
* If I calculated the score for all 80 classes in box 1, and found that the score for the car class (class 3) is the maximum, I'll assign the score 0.44 and class "3" to this box "1".

#### Visualizing classes
A useful way to visualize what YOLO is predicting on an image:

- For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across the 80 classes, one maximum for each of the 5 anchor boxes).
- Color that grid cell according to what object that grid cell considers the most likely.

Doing this results in this picture: 

<img src="nb_images/proba_map.png" style="width:300px;height:300;">
<caption><center> <u><b>Figure 5</u></b>: Each one of the 19x19 grid cells is colored according to which class has the largest predicted probability in that cell.<br> </center></caption>

This visualization isn't a core part of the YOLO algorithm itself for making predictions; it's just a nice way of visualizing an intermediate result of the algorithm. 

#### Visualizing bounding boxes
Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:  

<img src="nb_images/anchor_map.png" style="width:200px;height:200;">
<caption><center> <u><b>Figure 6</u></b>: Each cell gives 5 boxes. In total, the model predicts: 19x19x5 = 1805 boxes just by looking once at the image (one forward pass through the network)! (Different colors denote different classes) <br> </center></caption>

#### Non-Max suppression
In the figure above, the only boxes plotted are ones for which the model had assigned a high probability, but these are still too many boxes. I'd like to reduce the algorithm's output to a much smaller number of detected objects.  

To do so, I'll use **non-max suppression**. Specifically, the algorithm carrys out these steps: 
- Get rid of boxes with a low score. Meaning, the box is not very confident about detecting a class, either due to the low probability of any object, or low probability of this particular class.
- Select only one box when several boxes overlap with each other and detect the same object. (an important step)

<a name='2-2'></a>
### 2.2 - Filtering with a Threshold on Class Scores

Firstly, I will apply a filter by thresholding to get rid of any box for which the class "score" is less than a chosen threshold. 

The model gives a total of 19x19x5x85 numbers, with each box described by 85 numbers. It's convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:  
- `box_confidence`: tensor of shape $(19, 19, 5, 1)$ containing $p_c$ (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
- `boxes`: tensor of shape $(19, 19, 5, 4)$ containing the midpoint and dimensions $(b_x, b_y, b_h, b_w)$ for each of the 5 boxes in each cell.
- `box_class_probs`: tensor of shape $(19, 19, 5, 80)$ containing the "class probabilities" $(c_1, c_2, ... c_{80})$ for each of the 80 classes for each of the 5 boxes per cell.

<a name='-'></a>
### yolo_filter_boxes

The function `yolo_filter_boxes()` does the follwing:
```
1. Compute box scores by doing the elementwise product as described in Figure 4 ($p \times c$).  
```
```
**broadcasting** is used here (multiplying vectors of different sizes).

2. For each box, find:
    - the index of the class with the maximum box score
    - the corresponding box score

* Applying `reduce_max` normally collapses the axis for which the maximum is applied.  `keepdims=False` 
No need to keep the last dimension after applying the maximum here.


3. Create a mask by using a threshold. The mask should be `True` for the boxes to keep. 

4. Use TensorFlow to apply the mask to `box_class_scores`, `boxes` and `box_classes` to filter out the boxes that are not wanted. only the subset of boxes to keep will remain.   


```python
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs 

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    # IMPORTANT: set axis to -1 (last one)
    box_classes = tf.math.argmax(box_scores,axis = -1)
    box_class_scores = tf.math.reduce_max(box_scores, axis = -1)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes to keep (with probability >= threshold)
    filtering_mask = (box_class_scores >= threshold)
    
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes
```

<a name='2-3'></a>
### 2.3 - Non-max Suppression

Even after filtering by thresholding over the class scores, the model will end up with a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS). 

<img src="nb_images/non-max-suppression.png" style="width:500px;height:400;">
<caption><center> <u> <b>Figure 7</b> </u>: In this example, the model has predicted 3 cars, but it's actually 3 predictions of the same car. Running non-max suppression (NMS) will select only the most accurate (highest probability) of the 3 boxes. <br> </center></caption>


Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU.
<img src="nb_images/iou.png" style="width:500px;height:400;">
<caption><center> <u> <b>Figure 8</b> </u>: Definition of "Intersection over Union". <br> </center></caption>

<a name='-'></a>
###  iou

Implement `iou()` 

- This function uses the convention that (0,0) is the top-left corner of an image, (1,0) is the upper-right corner, and (1,1) is the lower-right corner. In other words, the (0,0) origin starts at the top left corner of the image. As x increases, move to the right.  As y increases, move down.
- For this function, a box is defined using its two corners: upper left $(x_1, y_1)$ and lower right $(x_2,y_2)$, instead of using the midpoint, height and width. This makes it a bit easier to calculate the intersection.
- To calculate the area of a rectangle, multiply its height $(y_2 - y_1)$ by its width $(x_2 - x_1)$. Since $(x_1,y_1)$ is the top left and $x_2,y_2$ are the bottom right, these differences should be non-negative.
- finding the **intersection** of the two boxes $(xi_{1}, yi_{1}, xi_{2}, yi_{2})$: 
    - The top left corner of the intersection $(xi_{1}, yi_{1})$ is found by comparing the top left corners $(x_1, y_1)$ of the two boxes and finding a vertex that has an x-coordinate that is closer to the right, and y-coordinate that is closer to the bottom.
    - The bottom right corner of the intersection $(xi_{2}, yi_{2})$ is found by comparing the bottom right corners $(x_2,y_2)$ of the two boxes and finding a vertex whose x-coordinate is closer to the left, and the y-coordinate that is closer to the top.
    - The two boxes **may have no intersection**.   if the intersection coordinates end up being the top right and/or bottom left corners of an intersection box.  Another way to think of this is if you calculate the height $(y_2 - y_1)$ or width $(x_2 - x_1)$ and find that at least one of these lengths is negative, then there is no intersection (intersection area is zero).  
    - The two boxes may intersect at the **edges or vertices**, in which case the intersection area is still zero.  This happens when either the height or width (or both) of the calculated intersection is zero.


```python
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """


    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    ### START CODE HERE
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    inter_width = xi2 - xi1
    inter_height =  yi2 - yi1
    inter_area = max(inter_width,0) * max(inter_height,0)
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] -box1[0]) * (box1[3] -box1[1])
    box2_area = (box2[2] -box2[0]) * (box2[3] -box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area    
    return iou
```

<a name='2-4'></a>
### 2.4 - YOLO Non-max Suppression

To implement non-max suppression. The key steps are: 
1. Select the box that has the highest score.
2. Compute the overlap of this box with all other boxes, and remove boxes that overlap significantly (iou >= `iou_threshold`).
3. Go back to step 1 and iterate until there are no more boxes with a lower score than the currently selected box.

This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.

<a name='-'></a>
### yolo_non_max_suppression

Implement `yolo_non_max_suppression()` using TensorFlow. TensorFlow has built-in functions that are used to implement non-max suppression (no need to use my own `iou()` implementation):

**Reference documentation**: 

- [tf.image.non_max_suppression()](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
```
tf.image.non_max_suppression(
    boxes,
    scores,
    max_output_size,
    iou_threshold=0.5,
    name=None
)
```

- [tf.gather()](https://www.tensorflow.org/api_docs/python/tf/gather)
```
keras.gather(
    reference,
    indices
)
```


```python
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None, ), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Also, this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes)
    
    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    return scores, boxes, classes
```

<a name='2-5'></a>
### 2.5 - Wrapping Up the Filtering

Now, I'll implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering through all the boxes using the functions previously implemented. 

<a name='-'></a>
### yolo_eval

This function `yolo_eval()` takes the output of the YOLO encoding and filters the boxes using score threshold and NMS.

Note: There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. YOLO converts between a few such formats at different times, using the following functions: 

```python
boxes = yolo_boxes_to_corners(box_xy, box_wh) 
```
which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of `yolo_filter_boxes`
```python
boxes = scale_boxes(boxes, image_shape)
```
YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image -- for example, the car detection dataset had 720x1280 images -- this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.  


```python
def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])
```


```python
def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model (≈1 line)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    # Perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)
    
    # Perform Non-max suppression with maximum number of boxes set to max_boxes and a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5)

    return scores, boxes, classes
```

<a name='3'></a>
## 3 - Test YOLO Pre-trained Model on Images

Now, I'll use a pre-trained model and test it on the car detection dataset.  

<a name='3-1'></a>
### 3.1 - Defining Classes, Anchors and Image Shape

The objective is to detect 80 classes, using 5 anchor boxes. The information on the 80 classes and 5 boxes is gathered in two files: "coco_classes.txt" and "yolo_anchors.txt". The car detection dataset has 720x1280 images, which are pre-processed into 608x608 images.


```python
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
model_image_size = (608, 608) # Same as yolo_model input layer size
```

<a name='3-2'></a>
### 3.2 - Loading a Pre-trained Model

Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. You are going to load an existing pre-trained Keras YOLO model stored in "yolo.h5". These weights come from the official YOLO website, and were converted using a function written by Allan Zelener. References are at the end of this notebook. 


```python
# load the model from this file.
yolo_model = load_model("model_data/", compile=False)
```


```python
# This loads the weights of a trained YOLO model. 
# A summary of the layers your model contains:
yolo_model.summary()
```

    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 608, 608, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 608, 608, 32) 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 608, 608, 32) 128         conv2d[0][0]                     
    __________________________________________________________________________________________________
    leaky_re_lu (LeakyReLU)         (None, 608, 608, 32) 0           batch_normalization[0][0]        
    __________________________________________________________________________________________________
    max_pooling2d (MaxPooling2D)    (None, 304, 304, 32) 0           leaky_re_lu[0][0]                
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 304, 304, 64) 18432       max_pooling2d[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 304, 304, 64) 256         conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)       (None, 304, 304, 64) 0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 152, 152, 64) 0           leaky_re_lu_1[0][0]              
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 152, 152, 128 73728       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 152, 152, 128 512         conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)       (None, 152, 152, 128 0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 152, 152, 64) 8192        leaky_re_lu_2[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 152, 152, 64) 256         conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)       (None, 152, 152, 64) 0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 152, 152, 128 73728       leaky_re_lu_3[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 152, 152, 128 512         conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)       (None, 152, 152, 128 0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, 76, 76, 128)  0           leaky_re_lu_4[0][0]              
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 76, 76, 256)  294912      max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 76, 76, 256)  1024        conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_5 (LeakyReLU)       (None, 76, 76, 256)  0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 76, 76, 128)  32768       leaky_re_lu_5[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 76, 76, 128)  512         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)       (None, 76, 76, 128)  0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 76, 76, 256)  294912      leaky_re_lu_6[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 76, 76, 256)  1024        conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)       (None, 76, 76, 256)  0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, 38, 38, 256)  0           leaky_re_lu_7[0][0]              
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 38, 38, 512)  1179648     max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, 38, 38, 512)  2048        conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)       (None, 38, 38, 512)  0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 38, 38, 256)  131072      leaky_re_lu_8[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, 38, 38, 256)  1024        conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)       (None, 38, 38, 256)  0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 38, 38, 512)  1179648     leaky_re_lu_9[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, 38, 38, 512)  2048        conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_10 (LeakyReLU)      (None, 38, 38, 512)  0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 38, 38, 256)  131072      leaky_re_lu_10[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 38, 38, 256)  1024        conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_11 (LeakyReLU)      (None, 38, 38, 256)  0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 38, 38, 512)  1179648     leaky_re_lu_11[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 38, 38, 512)  2048        conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)      (None, 38, 38, 512)  0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)  (None, 19, 19, 512)  0           leaky_re_lu_12[0][0]             
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 19, 19, 1024) 4718592     max_pooling2d_4[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 19, 19, 1024) 4096        conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_13 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 19, 19, 512)  524288      leaky_re_lu_13[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 19, 19, 512)  2048        conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_14 (LeakyReLU)      (None, 19, 19, 512)  0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 19, 19, 1024) 4718592     leaky_re_lu_14[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 19, 19, 1024) 4096        conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_15 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 19, 19, 512)  524288      leaky_re_lu_15[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 19, 19, 512)  2048        conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_16 (LeakyReLU)      (None, 19, 19, 512)  0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 19, 19, 1024) 4718592     leaky_re_lu_16[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 19, 19, 1024) 4096        conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_17 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 19, 19, 1024) 9437184     leaky_re_lu_17[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 19, 19, 1024) 4096        conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 38, 38, 64)   32768       leaky_re_lu_12[0][0]             
    __________________________________________________________________________________________________
    leaky_re_lu_18 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, 38, 38, 64)   256         conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 19, 19, 1024) 9437184     leaky_re_lu_18[0][0]             
    __________________________________________________________________________________________________
    leaky_re_lu_20 (LeakyReLU)      (None, 38, 38, 64)   0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, 19, 19, 1024) 4096        conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    space_to_depth_x2 (Lambda)      (None, 19, 19, 256)  0           leaky_re_lu_20[0][0]             
    __________________________________________________________________________________________________
    leaky_re_lu_19 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 19, 19, 1280) 0           space_to_depth_x2[0][0]          
                                                                     leaky_re_lu_19[0][0]             
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 19, 19, 1024) 11796480    concatenate[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, 19, 19, 1024) 4096        conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_21 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 19, 19, 425)  435625      leaky_re_lu_21[0][0]             
    ==================================================================================================
    Total params: 50,983,561
    Trainable params: 50,962,889
    Non-trainable params: 20,672
    __________________________________________________________________________________________________
    

This model converts a preprocessed batch of input images (shape: (m, 608, 608, 3)) into a tensor of shape (m, 19, 19, 5, 85) as explained in Figure (2) above.

<a name='3-3'></a>
### 3.3 - Convert Output of the Model to Usable Bounding Box Tensors

The output of `yolo_model` is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion. You will need to call `yolo_head` to format the encoding of the model you got from `yolo_model` into something decipherable:

yolo_model_outputs = yolo_model(image_data) 

yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))

The variable `yolo_outputs` will be defined as a set of 4 tensors used as input by yolo_eval function. 

Note: yolo_head implementation details, can be found in this file `keras_yolo.py`.

<a name='3-4'></a>
### 3.4 - Filtering Boxes

`yolo_outputs` gives  all the predicted boxes of `yolo_model` in the correct format. To perform filtering and select only the best boxes, I will call `yolo_eval`. to do so:

    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

<a name='3-5'></a>
### 3.5 - Run the YOLO on an Image

the process can be summarized as follows:

`yolo_model.input` is given to `yolo_model`. The model is used to compute the output `yolo_model.output`

`yolo_model.output` is processed by `yolo_head`. 
It gives  `yolo_outputs`
`yolo_outputs` goes through a filtering function, `yolo_eval`. 
It then outputs the predictions: `out_scores`, `out_boxes`, `out_classes`.

`predict(image_file)` function runs the graph to test YOLO on an image to compute `out_scores`, `out_boxes`, `out_classes`.

which opens the image file and scales, reshapes and normalizes the image. It returns the outputs:

    image: a python (PIL) representation of your image used for drawing boxes.
    image_data: a numpy-array representing the image. This will be the input to the CNN.


```python
def predict(image_file):
    """
    Runs the graph to predict boxes for "image_file". Prints and plots the predictions.
    
    Arguments:
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    
    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
    # Generate colors for drawing bounding boxes.
    colors = get_colors_for_classes(len(class_names))
    
    # Draw bounding boxes on the image file
    #draw_boxes2(image, out_scores, out_boxes, out_classes, class_names, colors, image_shape)
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=100)
    # Display the results in the notebook
    output_image = Image.open(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes
```

Running the following cell on the "test.jpg" image to verify that the function works well.


```python
out_scores, out_boxes, out_classes = predict("test.jpg")
```

    Found 10 boxes for images/test.jpg
    car 0.89 (367, 300) (745, 648)
    car 0.80 (761, 282) (942, 412)
    car 0.74 (159, 303) (346, 440)
    car 0.70 (947, 324) (1280, 705)
    bus 0.67 (5, 266) (220, 407)
    car 0.66 (706, 279) (786, 350)
    car 0.60 (925, 285) (1045, 374)
    car 0.44 (336, 296) (378, 335)
    car 0.37 (965, 273) (1022, 292)
    traffic light 0.36 (681, 195) (692, 214)
    


    
![png](output_34_1.png)
    


This model is actually able to detect 80 different classes listed in "coco_classes.txt". 

Feel free to test the model on your own images:

If you were to run a session in a for loop over all images. you would get this:

<center>
<video width="400" height="200" src="nb_images/pred_video_compressed2.mp4" type="video/mp4" controls>
</video>
</center>

<caption><center> Predictions of the YOLO model on pictures taken from a camera while driving around the Silicon Valley <br> 

<a name='4'></a>
## 4 - Summary for YOLO

- Input image (608, 608, 3)
- The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. 
- After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
    - Each cell in a 19x19 grid over the input image gives 425 numbers. 
    - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture. 
    - 85 = 5 + 80 where 5 is because $(p_c, b_x, b_y, b_h, b_w)$ has 5 numbers, and 80 is the number of classes we'd like to detect
- Then, select only few boxes based on:
    - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
    - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
- This gives you YOLO's final output. 

<font color='blue'>
    
**Final Notes**:
    
- YOLO is a state-of-the-art object detection model that is fast and accurate
- It runs an input image through a CNN, which outputs a 19x19x5x85 dimensional volume. 
- The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
- Filter through all the boxes using non-max suppression. Specifically: 
    - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
    - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
- Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, previously trained model parameters were used in this project.

**Revisiting Project Objectives!** 
- Detected objects in a car detection dataset
- Implemented non-max suppression to achieve better accuracy
- Implemented intersection over union as a function of NMS
- Created usable bounding box tensors from the model's predictions

<a name='5'></a>
## 5 - References

The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's GitHub repository. The pre-trained weights used in this exercise came from the official YOLO website. 
- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
- Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
- Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
- The official YOLO website (https://pjreddie.com/darknet/yolo/) 
- Coursera and DeepLearning.ai (https://www.deeplearning.ai/) 

### Car detection dataset

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The Drive.ai Sample Dataset</span> (provided by drive.ai) is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. Thanks to Brody Huval, Chih Hu and Rahul Patel for  providing this data. 

## 🛠 Skills Used
Python, Deep Learning, Optimization, Object Detection, YOLO algorithm, Non-maximum Suppression


## 🚀 About Me
👋 Hi, I’m @Raed-Alshehri

👀 I’m interested in data science, machine learning, and statistics.

🌱 I’m applying my skills in the data analytics field using Python, R, and SQL


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://raed-alshehri.github.io/RaedAlshehri.github.io/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/raedalshehri/)


## Feedback

If you have any feedback, please reach out to me at alshehri.raeda@gmail.com

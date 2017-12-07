# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:47:24 2017

"""



import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches

#import operator

import os

def get_intersection(p1,p2):
  x11,y11,x12,y12 = p1
  x21,y21,x22,y22 = p2
  left = max(x11, x21)
  right = min(x12, x22)
  top = max(y11, y21)
  bottom = min(y12, y22)
  if (left>right) or (top>bottom):
    res = 0
  else:
    res = (right-left) * (bottom - top)
  return res

def get_area(p):
  return (p[2]-p[0])*(p[3]-p[1])



def get_IoU(p1,p2):
  v_inter = get_intersection(p1,p2)
  v_union = get_area(p1) + get_area(p2) - v_inter
  return v_inter/v_union
  
def check_empty_margin(box, np_image):
  if len(np_image.shape)>2:
    h = np_image.shape[1]
    w = np_image.shape[2]
    np_image = np_image.reshape((h,w))
  x1,y1,x2,y2=box
  res = np.sum(np_image[y1,x1:x2])
  res+= np.sum(np_image[y2-1,x1:x2])
  res+= np.sum(np_image[y1:y2,x1])
  res+= np.sum(np_image[y1:y2,x2-1])
  return res==0

def simple_analysis(np_image, np_preds, iou_threshold, score_threshold):
  pred_mat = np.zeros((np_preds.shape[1],np_preds.shape[2]), dtype=int)
  objects = {x : list() for x in range(10)}
  boxes = [] # list of touple (proba,valid,label,x1,y1,x2,y2)

  # generate labels/boxes - INEFFICIENT IMPLEMENTATION
  for h in range(np_preds.shape[1]):
    for w in range(np_preds.shape[2]):
      argmax = int(np.argmax(np_preds[0,h,w,:]))
      proba = np_preds[0,h,w,argmax]
      pred_mat[h,w] = -1
      if  (proba > score_threshold):
        pred_mat[h,w] = argmax
        c_rect = (w*2,h*2,w*2+orig_w,h*2+orig_h)
        is_centered = check_empty_margin(c_rect, np_image)
        x1,y1,x2,y2 = c_rect
        objects[argmax].append((proba,c_rect))
        if is_centered:
          boxes.append([proba, 1, argmax, x1, y1, x2, y2])
        else:
          boxes.append([proba, 0, argmax, x1, y1, x2, y2])

  #now two stage non-max-suppression with IoU thresholding
  
  # 1st phase apply non-max suppression only same labels with IoU > thres
  #  - INEFFICIENT IMPLEMENTATION
  for box in boxes:
    p1 = box[3:]
    label1 = box[2]
    for inter_box in boxes:
      if (box != inter_box) and (box[1] != 0) and (inter_box [1] != 0):
        label2 = inter_box[2]
        iou = get_IoU(p1,inter_box[3:])
        if (iou > iou_threshold) and (label1==label2):
          if (box[0] >= inter_box[0]):
            inter_box[1] = 0
          else:
            box[1] = 0
  
  # 2nd phase apply straight non-max suppression over rects with IoU > thres
  #  - INEFFICIENT IMPLEMENTATION
  for box in boxes:
    p1 = box[3:]
    for inter_box in boxes:
      if (box != inter_box) and (box[1] != 0) and (inter_box [1] != 0):
        iou = get_IoU(p1,inter_box[3:])
        if (iou > iou_threshold):
          if (box[0] >= inter_box[0]):
            inter_box[1] = 0
          else:
            box[1] = 0
  # compile results
  final_boxes = []
  final_classes = []
  final_scores = []
  for box in boxes:
    if box[1] != 0:
      final_boxes.append(box[3:])
      final_classes.append(box[2])
      final_scores.append(box[0])
  return final_boxes, final_classes, final_scores, objects, boxes
  

def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    logger_lib = None
    print("Logger library not found in shared repo.", flush = True)
    #raise Exception("Couldn't find google drive folder!")
  else:  
    utils_path = os.path.join(drive_path, "_pyutils")
    print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
    logger_lib = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
    print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return logger_lib

class SimpleLogger:
  def __init__(self):
    self._outp_dir = "."
    return
  def VerboseLog(self, _str, show_time):
    print(_str, flush = True)

def LoadLogger(lib_name, config_file):
  module = load_module("logger", "logger.py")
  if module is not None:
    logger = module.Logger(lib_name = lib_name, config_file = config_file)
  else:
    logger = SimpleLogger()
  return logger


if __name__=='__main__':
  
  FULL_DEBUG = False

  K = tf.keras.backend
  
  K.clear_session()
  
  logger = LoadLogger(lib_name = "MFCN", config_file = "config.txt")
  
  saved_model_file = os.path.join(logger._outp_dir,'full_conv.h5')
  saved_dag_file = os.path.join(logger._outp_dir,'full_dag.h5')
  
  
  logger.VerboseLog("Loading dataset...")
  
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
  logger.VerboseLog("Done loading dataset.", show_time=True)
  
  
  X_train = X_train.reshape((-1,28,28,1)) / 255
  X_test = X_test.reshape((-1,28,28,1)) / 255
  
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10).reshape((-1,1,1,10))
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10).reshape((-1,1,1,10))

  final_kernel_size = 14

  logger.VerboseLog("Preparing test scene...")
  test_size = 100
  new_image = np.zeros((test_size,test_size))
  
  nr_images = 10
  
  indices = np.arange(X_test.shape[0])

  orig_h, orig_w = (28,28)
  
  sampled = np.random.choice(indices,nr_images, replace=False)
  labels = []
  pos_w = np.random.randint(1,10)
  pos_h = np.random.randint(1,10)
  for i in range(nr_images):
    ind = sampled[i]
    x_cur = X_test[ind]
    if (pos_h<=(test_size-orig_h)):
      new_image[pos_h:(pos_h+orig_h),pos_w:(pos_w+orig_w)] = x_cur.reshape((orig_h,orig_w))
      labels.append(np.argmax(y_test[ind]))
      pos_w += 28 + np.random.randint(-5,10)
    if pos_w >= (test_size - orig_w):
      pos_w = np.random.randint(1,10)
      pos_h += 28 + np.random.randint(-3,8)
  plt.matshow(new_image,cmap="gray")
  plt.show()
  
  pred_H = int(new_image.shape[0] / 2 - final_kernel_size + 1)
  pred_W = int(new_image.shape[1] / 2 - final_kernel_size + 1)
  np_boxes = np.zeros((pred_H,pred_W,4))
  for h in range(pred_H):
    for w in range(pred_W):
      np_boxes[h,w] = (h*2, w*2, h*2 + orig_h, w*2 + orig_w)

  np_boxes = np_boxes.reshape((1,pred_H,pred_W,4))
  min_threshold = 0.98
  logger.VerboseLog("Done preparing test scene.", show_time = True)

  create_dag_required = True
  
  if os.path.isfile(saved_dag_file):
    logger.VerboseLog("Loading DAG model...")
    final_model = tf.keras.models.load_model(saved_dag_file)
    logger.VerboseLog("Done loading DAG model.", show_time = True)
    create_dag_required = False
  elif os.path.isfile(saved_model_file):
    logger.VerboseLog("Loading model...")
    model = tf.keras.models.load_model(saved_model_file)
    logger.VerboseLog("Done loading model.", show_time=True)
    logger.LogKerasModel(model)    
  else:
    logger.VerboseLog("Defining and training model...")
    X_input = tf.keras.layers.Input((None,None,1))
    X = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same')(X_input)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Dropout(rate = 0.1)(X)
    
    X = tf.keras.layers.MaxPooling2D((2,2))(X)
    
    X = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Dropout(rate = 0.2)(X)
    
    #X = MaxPooling2D((2,2))(X)
    
    #X = Conv2D(filters = 128, kernel_size = 3, padding = 'same')(X)
    #X = BatchNormalization(axis = 3)(X)
    #X = Activation('relu')(X)
    
    X = tf.keras.layers.Conv2D(filters = 10, 
                               kernel_size = (final_kernel_size,final_kernel_size), 
                               activation = 'softmax')(X)
    
    model = tf.keras.models.Model(inputs = X_input, outputs = X)
    
    logger.LogKerasModel(model)
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    model.fit(X_train,y_train, epochs = 10, batch_size = 128, 
              validation_data = (X_test, y_test),
              callbacks = [
                           logger.KerasEpochCallback(),
                           logger.KerasTensorboardCallback()
                          ])
    
    model.save(saved_model_file)
  

  model_output = model.outputs[0]
  model_input = model.inputs[0]

  
  ###
  ### now the actual model ops
  ###
  
  if create_dag_required:
    sess = K.get_session() # sess = tf.Session() 
    
    logger.VerboseLog("Creating final DAG...")

    
    def layer_generator(input_tensor):   
      max_boxes = 10
      print("Lambda Layer function start...")
      tf_max_boxes_tensor = K.variable(max_boxes, dtype='int32') 
      print(" Input tensor: {}".format(input_tensor))
    
      # create boxes template tensor
      tf_all_boxes = K.variable(np_boxes)
      # 1st initialize max_box tensor and boxes template tensor
      print(" Pre init")
      sess.run(tf.variables_initializer([tf_all_boxes,tf_max_boxes_tensor])) 
      print(" Post init")
      # get max score for output volume
      tf_slide_scores = K.max(input_tensor, axis = 3)
      # get class for each output class
      tf_slide_classes = K.argmax(input_tensor, axis = 3)
      # create mask - get only scores above threshold
      tf_slide_masks = tf_slide_scores >= min_threshold
      
      # get tensor (1d) of scores out of whole image based on mask
      tf_scores = tf.boolean_mask(tf_slide_scores, tf_slide_masks)
      # get tensor (1d-of-4f=2d )of boxes out of whole image based on mask
      tf_boxes = tf.boolean_mask(tf_all_boxes, tf_slide_masks)
      # get tensor of classes out of whole image based on mask
      tf_classes = tf.boolean_mask(tf_slide_classes, tf_slide_masks)
      
      # run non-max-suppression with IoU 0.5
      # all overlapping (more than 0.5) will be dropped based on lower score
      tf_select = tf.image.non_max_suppression(tf_boxes, tf_scores,
                                               tf_max_boxes_tensor,
                                               iou_threshold = 0.4)
      # get only "winning" indices from previous operation
      tf_final_classes = K.cast(tf.gather(tf_classes, tf_select), dtype = 'float32')
      tf_final_boxes = tf.gather(tf_boxes, tf_select)
      tf_final_scores = tf.gather(tf_scores, tf_select)
      boxes_shape = K.int_shape(tf_final_boxes)
      print(' Boxes tensor shape: {}'.format(boxes_shape))
      tf_final_boxes_unrolled = K.reshape(tf_final_boxes, (-1,))
      
      # now prepare a long tensor of size OBJECTS+OBJECTS+OBJECTS*4 = OBJECTS * 6
      tf_output = K.concatenate([tf_final_classes, tf_final_scores, tf_final_boxes_unrolled],
                                axis = 0)
      print("Lambda Layer function end.")
      return tf_output
    
    final_dag_layer = tf.keras.layers.Lambda(layer_generator)(model_output)
    
    final_model = tf.keras.models.Model(inputs = model_input, 
                                        outputs = final_dag_layer)
    logger.VerboseLog("Saving DAG model...")
    final_model.save(saved_dag_file)
    logger.VerboseLog("Done saving DAG model.")
    
      
    
    
    logger.VerboseLog("Done creating final DAG.", show_time = True)


  ###  
  ### sess.run obs:
  ###  when a model uses BatchNorm additional placeholder is needed 
  ###  in the feed_dict {K.learning_phase(): 0}.
  ###
  ### conv phase (simple):  0.15s - 0.20s / A: 0.05 - 0.07
  ### box phase (ineff):    0.16s - 0.40s / A: 0.05 - 0.07
  ### total time for v1:    0.30s - 0.60s / A: 0.10 - 0.14
  ### graph predict v2:     0.30s - 0.35s / A: 0.08 - 0.09 (-20% to -40%)
  
  GRAPH_INFERENCE = True
  objects = None
  
  if not GRAPH_INFERENCE:
    logger.VerboseLog("Running prediction...")
    pred = sess.run(model_output, 
                    feed_dict = { model_input : new_image.reshape((1,test_size,test_size,1)),
                                  K.learning_phase() : 0})
    logger.VerboseLog("Done running prediction. Output volume: {}.".format(pred.shape), show_time = True)
    logger.VerboseLog("Running NMS / IoU /etc ...")
    final_boxes, classes, scores, objects, boxes = simple_analysis(new_image, pred, 
                                    iou_threshold = 0.2,
                                    score_threshold = min_threshold)
    logger.VerboseLog("Done running NMS / IoU /etc.", show_time = True)
  else:
    logger.VerboseLog("Running full graph inference...")
    #final_boxes, classes, scores = sess.run([tf_final_boxes,
    #                                         tf_final_classes,
    #                                         tf_final_scores],
    #                                         feed_dict = 
    #                                         { model_input : new_image.reshape((1,test_size,test_size,1)),
    #                                           K.learning_phase() : 0})
    
    final_model_output = final_model.outputs[0]
    preds = sess.run(final_model_output, 
                     feed_dict ={ model_input : new_image.reshape((1,test_size,test_size,1)),
                                  K.learning_phase() : 0})
    nr_preds = int(preds.shape[0] / 6)
    classes = preds[:nr_preds]
    scores = preds[nr_preds:nr_preds*2]
    final_boxes = (preds[nr_preds*2:]).reshape((-1,4))
    logger.VerboseLog("Done running full graph inference.", show_time = True)
  #pred = model.predict(new_image.reshape((1,100,100,1)))  
  
  
  
  
  
      

  np.set_printoptions(linewidth = 180)

 
  logger.VerboseLog("Display phase ...")
  
  if objects != None:
    #show results
    fig,ax = plt.subplots(1) #, figsize=(test_size,test_size))
    ax.imshow(new_image, cmap='gray')
    counts = {i:0 for i in range(0)}
    for label,r_list in objects.items():
      if label in labels:
        clr = 'g'
      else:
        clr = 'r'
      counts[label] = len(r_list)
      for (proba,r) in r_list:
        patch = patches.Rectangle((r[0],r[1]),
                                  r[2]-r[0], r[3]-r[1],
                                  linewidth=1,
                                  edgecolor=clr,
                                  facecolor='none')
        ax.add_patch(patch)
    plt.show()
    for w in sorted(counts, key=counts.get, reverse=True):
      logger.VerboseLog("Class: {} Nr.Rects: {}".format(w,counts[w]))



  fig,ax = plt.subplots(1) #, figsize=(test_size,test_size))
  ax.imshow(new_image, cmap='gray')
  counts = {i:0 for i in range(0)}
  for i,box in enumerate(final_boxes):
    label = classes[i]
    proba = scores[i]
    r = box
    if label in labels:
      clr = 'g'
    else:
      clr = 'r'
    if GRAPH_INFERENCE:
      y1 = r[0]
      x1 = r[1]
      y2 = r[2]
      x2 = r[3]
    else:
      x1 = r[0]
      y1 = r[1]
      x2 = r[2]
      y2 = r[3]      
    patch = patches.Rectangle((x1,y1),
                              x2-x1, y2-y1,
                              linewidth=1,
                              edgecolor=clr,
                              facecolor='none')
    ax.add_patch(patch)
    ax.text(x1, y1-1, "{}:{:.5f}".format(label,proba), color=clr) 
  
  plt.show()

  logger.VerboseLog("Done display phase.", show_time = True)
    
  logger.VerboseLog("Final boxes:")
  for i,box in enumerate(final_boxes):
    logger.VerboseLog("  Class:{} Proba: {:.5f} Box: {}".format(
        classes[i],scores[i],box))
    
  if FULL_DEBUG and (boxes != None):  
    logger.VerboseLog("100% boxes:")
    for box in boxes:
      if box[0]==1:
        logger.VerboseLog(" {}".format(box))
        
  #logger.VerboseLog("Clossing session...")
  #sess.close()
  #logger.VerboseLog("Done clossing session.", show_time = True)

  
          
  

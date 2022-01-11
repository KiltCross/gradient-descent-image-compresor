# -*- coding: utf-8 -*-
"""main

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1M0eWI9JNKnSLHf0CCbrDHTC4OA_VSr2R
"""
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import numpy as np
import math

import time


def metadata(desire_data, desire_data_of_desire_data=False):
  switcher = {
      'image_shape': [2**7 , 2**6 , 4], #shape of the images ({2**7,2**6,4}={128,64,4})
      'mini_batch_size': 10, #size of the mini batches what the images are diveded
      'batch_size': 24*2, #total of mini batches. (24 for recolor and 2 for swamp)
      'epoch': 1000, #epochs
      'train_batch_size': 1, #quantity of mini batches used to do the training
      'test_batch_size': 1, #quantity of mini batches used to do the testing
      'degree':  4, #the max number that X is elevate to
      'number_of_theta_of_same_X' : 1# the number theta that multiply the same X
  }
  return switcher.get(desire_data)[desire_data_of_desire_data] if desire_data_of_desire_data else switcher.get(desire_data)

def image_resize(img, new_shape=[128,64]):
  new_shape = new_shape[:2]
  new_shape[0] , new_shape[1] = new_shape[1], new_shape[0]
  img = Image.fromarray((img*255).astype(np.uint8),"RGBA")
  img = img.resize(new_shape)
  img = np.array(img)/255
  return img

#the ds is divided on groups of 244 images, and the batch size is in units of this groups

def DS(desired_ds='train', batch_size=2*24, train_batch_size = 1, test_batch_size = 1, mini_batch_size=244, image_shape=[128,64,4]):

  train_batch_size = int(train_batch_size * mini_batch_size)
  test_batch_size = int(test_batch_size * mini_batch_size)

  last_timing = time.time()
  time_elapsed = 0
  count = 0

  if (desired_ds == 'train'):
    print("Importing main training dataset")
    ds = np.zeros([train_batch_size,image_shape[0],image_shape[1],image_shape[2]], dtype='float32')
    for train_image in range(train_batch_size):

      start = time.time()
      count += 1
      time_left = (time_elapsed/count)*(train_batch_size-count)

      ds[train_image] =  image_resize(mpimg.imread("Data_set/"+str(train_image+1)+".png"),image_shape[:2])
      time_elapsed += (start - last_timing)
      print("Image number: "+str(train_image+1)+"/"+str(train_batch_size)+"  |  Time left: "+str(int(time_left/3512))+" hours - "+str(int((time_left%3512)/60))+" minutes - "+str(int(time_left%60))+" seconds  |  Time elapsed: "+str(int(time_elapsed/(60*60)))+" hours - "+str(int(time_elapsed%(60*60)/60))+" minutes - "+str(int(time_elapsed%60))+" seconds")
      
      last_timing = start


    ds = np.reshape(ds, [train_batch_size,image_shape[0],image_shape[1],image_shape[2]])
    ds[:,:,:,3] = 1
    np.random.shuffle(ds)

  if (desired_ds == 'test'):
    print("Importing test dataset")
    ds = np.zeros([test_batch_size,image_shape[0],image_shape[1],image_shape[2]], dtype='float32')
    for test_image in range(test_batch_size):

      start = time.time()
      count += 1
      time_left = (time_elapsed/count)*(test_batch_size-count)

      ds[test_image] = image_resize(mpimg.imread("Data_set/"+str(train_batch_size + test_image+1)+".png"),image_shape[:2])
      time_elapsed += (start - last_timing)
      print("Image number: "+str(test_image+1)+"/"+str(test_batch_size)+"  |  Time left: "+str(int(time_left/3512))+" hours - "+str(int((time_left%3512)/60))+" minutes - "+str(int(time_left%60))+" seconds  |  Time elapsed: "+str(int(time_elapsed/(60*60)))+" hours - "+str(int(time_elapsed%(60*60)/60))+" minutes - "+str(int(time_elapsed%60))+" seconds")

      last_timing = start      

    ds = np.reshape(ds, [test_batch_size,image_shape[0],image_shape[1],image_shape[2]])
    np.random.shuffle(ds)

  return ds

def optimizer(grad, theta):
  alpha = 0.02
  beta_1 = 0.999
  beta_2 = 0.999						#initialize the values of the parameters
  epsilon = 1e-8
  #def func(x):
    #return x*x -4*x + 4
  #def grad_func(x):					#calculates the gradient
    #return 2*x - 4
  #theta_0 = 0						#initialize the vector
  m_t = 0 
  v_t = 0 
  t = 0

  while (True):					#till it gets converged
    t+=1
    g_t = grad		#computes the gradient of the stochastic function
    m_t = beta_1*m_t+ (1-beta_1)*g_t	#updates the moving averages of the gradient
    v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
    m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
    v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates
    theta_prev = theta								
    theta = theta - (alpha*m_cap)/((v_cap**(1/2))+epsilon)	#updates the parameters
    #print(t)
    if(theta.all() == theta_prev.all()):		#checks if it is converged or not
      break
  return theta


def X_smoother_function(X, max_X, degree):
  return (X/max_X)**degree

def X_by_theta_unit_smoother_function(unit):
  if (unit > 1):
    #return 1
     return unit
  else:
    return unit

def decompressor_2(compressed_image, image_shape, the_degree, number_of_theta_of_same_X, give_decompress_image_original_sign=False):
  #print("decompressing the compressed image:") 
  if (len(compressed_image.shape) > 1): 
    compressed_image = compressed_image[0]
  #print([compressed_image,image_shape])
  #decompressed_image = tf.Variable(tf.zeros(image_shape))
  decompressed_image = np.zeros(image_shape)
  decompressed_image_original_sign = np.zeros(image_shape)
  #last_timing = time.time()
  #time_elapsed = 0
  #count = 0
  #the_degree = int((compressed_image.shape[0]-1)/3)
  total_iteration = image_shape[0] * image_shape[1] * image_shape[2] * the_degree
  for length in range(image_shape[0]):
    for width in range(image_shape[1]):
      for color_channel in range(image_shape[2]-1):
        for degree in range(the_degree):

            #start = time.time()
            h_length = X_by_theta_unit_smoother_function(compressed_image[degree]* X_smoother_function(length+1, image_shape[0], degree+1))
            h_length_r = X_by_theta_unit_smoother_function(compressed_image[(the_degree*3)+degree]* X_smoother_function(image_shape[0]-length, image_shape[0], degree+1))
            #print([length,width,color_channel,degree,h_length_r])
            h_width = X_by_theta_unit_smoother_function(compressed_image[the_degree+degree]* X_smoother_function(width+1, image_shape[1], degree+1))
            h_width_r = X_by_theta_unit_smoother_function(compressed_image[(the_degree*3)+the_degree+degree]* X_smoother_function(image_shape[1]-width, image_shape[1], degree+1))

            h_color_channel = X_by_theta_unit_smoother_function(compressed_image[(2*the_degree)+degree]*X_smoother_function(color_channel+1, image_shape[2], degree+1))
            h_color_channel_r = X_by_theta_unit_smoother_function(compressed_image[(the_degree*3)+(2*the_degree)+degree]* X_smoother_function(image_shape[2]-color_channel, image_shape[2], degree+1))

            decompressed_image[length,width,color_channel] += h_length + h_width + h_color_channel + h_length_r + h_width_r + h_color_channel_r
            del h_length
            del h_width
            del h_color_channel
            del h_length_r
            del h_width_r
            del h_color_channel_r

        decompressed_image[length,width,color_channel] += X_by_theta_unit_smoother_function(compressed_image[-1])
        if decompressed_image[length,width,color_channel] > 0: decompressed_image_original_sign[length,width,color_channel] = 1 
        if decompressed_image[length,width,color_channel] < 0: decompressed_image_original_sign[length,width,color_channel] = -1 
  decompressed_image[:,:,3] = (2*the_degree*3+1)

  return (abs(decompressed_image/(2*the_degree*3+1)) , decompressed_image_original_sign) if (give_decompress_image_original_sign) else abs(decompressed_image/(2*the_degree*3+1))

def loss_for_compressor_2(y_true, y_false):

  loss = (y_true - y_false)**2
  size = np.size(loss)
  loss = np.sum(loss)
  loss = loss / size

  return loss

def get_gradient(y_true, y_pred, size_of_theta,compressed_image, image_shape, the_degree, number_of_theta_of_same_X):
  #the_degree = int((size_of_theta-1)/3)
  gradient = np.zeros(size_of_theta)
  matrix = 2*(y_true - y_pred)
  #part_of_gradient = derivate(compressed_image, image_shape, the_degree, number_of_theta_of_same_X)
  #matrix *= 2*part_of_gradient
  #X = np.ones_like(y_true)
  
  for degree in range(the_degree):
    for lenght in range(np.shape(y_true)[0]):
      gradient[degree] += np.sum(matrix[lenght]*X_smoother_function(lenght+1, np.shape(y_true)[0], degree+1))
      gradient[(the_degree*3)+degree] += np.sum(matrix[lenght]*X_smoother_function(np.shape(y_true)[0]-lenght, np.shape(y_true)[0], degree+1))
      #print([lenght,degree,gradient[0]])
    for width in range(np.shape(y_true)[1]):  
      gradient[the_degree+degree] += np.sum(matrix[:,width]*X_smoother_function(width+1, np.shape(y_true)[1], degree+1))
      gradient[(the_degree*3)+the_degree+degree] += np.sum(matrix[:,width]*X_smoother_function(np.shape(y_true)[1]-width, np.shape(y_true)[1], degree+1))
    for color_channel in range(np.shape(y_true)[2]):  
      gradient[(2*the_degree)+degree] += np.sum(matrix[:,:,color_channel]*X_smoother_function(color_channel+1, np.shape(y_true)[2], degree+1))
      gradient[(the_degree*3)+(2*the_degree)+degree] += np.sum(matrix[:,:,color_channel]*X_smoother_function(np.shape(y_true)[2]-color_channel, np.shape(y_true)[2], degree+1))
    gradient[-1] = np.sum(matrix)
    gradient = np.divide(gradient,np.size(matrix))

  return gradient

def training_summary(epochs,actual_epoch,loss):
  if (actual_epoch == 0):
    #Setting the variables on the beginning of the loop
    global last_timing 
    global time_elapsed 
    global Iteration 
    global output_lines
    last_timing = time.time()
    time_elapsed = 0
    Iteration = 0
    output_lines = []
  start = time.time()
  Iteration += 1

  #The time calculations
  time_elapsed += (start - last_timing)
  time_left = (time_elapsed/Iteration)*((epochs)-Iteration)

  size_of_output_to_be_delete = np.size(output_lines)

  #Adding the actual training summery line
  print("Epoch: "+str(actual_epoch+1)+"/"+str(epochs)+"  |  Iteration duration: "+str(start - last_timing)+" seconds  |  Time left: "+str(int(time_left/3512))+" hours - "+str(int((time_left%3512)/60))+" minutes - "+str(int(time_left%60))+" seconds  |  Time elapsed: "+str(int(time_elapsed/(60*60)))+" hours - "+str(int(time_elapsed%(60*60)/60))+" minutes - "+str(int(time_elapsed%60))+" seconds | Loss: "+str(loss[Iteration-1]))
  """
  output_lines.append("Epoch: "+str(actual_epoch+1)+"/"+str(epochs)+"  |  Iteration duration: "+str(start - last_timing)+" seconds  |  Time left: "+str(int(time_left/3512))+" hours - "+str(int((time_left%3512)/60))+" minutes - "+str(int(time_left%60))+" seconds  |  Time elapsed: "+str(int(time_elapsed/(60*60)))+" hours - "+str(int(time_elapsed%(60*60)/60))+" minutes - "+str(int(time_elapsed%60))+" seconds | Loss: "+str(loss[Iteration-1]))

  #Delete the first line so the display isn't sturated
  if (len(output_lines) > 10):
    del output_lines[0]
  #Delete the previous summery
  size_of_output_to_be_delete = 0
  text_draft_text = "" #This backspace is for the line break made on every 'print' function.
  #This is for get the total lenght of the text.
  for i in range(len(output_lines)):
    size_of_output_to_be_delete += len(output_lines[i])
    text_draft_text = text_draft_text+"\010\010\010\010\010" #Two backspace for the parenthesis on the beginning and end of every line; 
                                                       #two backspace for the quotation marks, one after the parenthesis on the beginning and the other before the parenthesis on the end of every line;
                                                       #and one backspace for every linebreak.
  #This next three line delete the printed text
  for cell in range(size_of_output_to_be_delete):
    text_draft_text = text_draft_text+"\010"
  print(text_draft_text)
  for to_print in range(len(output_lines)):
    print(output_lines[to_print])

  #print(output_lines[])
  """        
  last_timing = start

  #Printing a special end line (on the end of the loop)
  if (actual_epoch == epochs):
    print("Finish!, take: "+str(int(time_elapsed/(60*60)))+":"+str(int(time_elapsed%(60*60)/60))+":"+str(int(time_elapsed%60)))
    del last_timing 
    del time_elapsed 
    del Iteration 
    del output_lines

def compressor_2(uncompress_image, compressed_image, the_degree, number_of_theta_of_same_X, epochs=10, alpha=0.001, return_decompressed_image=True):
  print("compressing image:")
  training_loss = np.zeros([epochs])

  last_timing = time.time()
  time_elapsed = 0
  count = 1
  total_iteration = uncompress_image.shape[0] * uncompress_image.shape[1] * uncompress_image.shape[2]  * epochs
  #decompressed_images_progress = []

  #loss = []
  #start = time.time()
  decompressed_image , decompressed_image_original_sign = decompressor_2(compressed_image,np.shape(uncompress_image), the_degree, number_of_theta_of_same_X, True)
  decompressed_image = Image.fromarray((decompressed_image*255).astype(np.uint8),"RGBA")
  #path_progress_images = "/media/manuel/sda21/Proyctos/Hypercompresor/Decompress_images_progress/"+str(epoch)+".jpg"
  path_progress_images = "Decompress_images_progress/"+str(0)+".png"
  decompressed_image.save(path_progress_images,format='png')
  for epoch in range(epochs):

    start = time.time()
    #decompressed_image = decompressor(compressed_image,np.shape(uncompress_image), the_degree, number_of_theta_of_same_X)
    grad = get_gradient(uncompress_image, decompressed_image*decompressed_image_original_sign, np.size(compressed_image),compressed_image,np.shape(uncompress_image), the_degree, number_of_theta_of_same_X)
    #print([epoch, decompressed_image[0,0,0],grad[0]])
    #print([epoch,compressed_image])
    compressed_image = optimizer(grad,compressed_image)
    #print([epoch,compressed_image])

    training_loss[epoch] = loss_for_compressor_2(uncompress_image, decompressed_image)
		    
    training_summary(epochs,epoch,training_loss)

    decompressed_image , decompressed_image_original_sign = decompressor_2(compressed_image,np.shape(uncompress_image), the_degree, number_of_theta_of_same_X, True)
    #decompressed_images_progress.append(decompressed_image)
    decompressed_image_to_save = Image.fromarray((decompressed_image*255).astype(np.uint8),"RGBA")
    #path_progress_images = "/media/manuel/sda21/Proyctos/Hypercompresor/Decompress_images_progress/"+str(epoch)+".jpg"
    path_progress_images = "Decompress_images_progress/"+str(epoch+1)+".png"
    decompressed_image_to_save.save(path_progress_images,format='png')

  np.save('training_loss',training_loss)
  np.save('compressed_image', compressed_image)
  #if (return_decompressed_image) else compressed_image , training_loss

#compressed_image = np.random.normal(size=[2*metadata("degree")*metadata('number_of_theta_of_same_X')*3+1])
compressed_image = np.random.normal(size=[2*metadata('degree')*3+1])*5
#compressed_image = np.zeros([2*metadata("degree")*3+1])+0.00001

decompressed_image = decompressor_2(compressed_image,[128,64,4],metadata("degree"),metadata('number_of_theta_of_same_X'))
#plt.imshow(decompressed_image)

train_ds=DS('train', train_batch_size=(metadata("train_batch_size")), mini_batch_size=metadata("mini_batch_size"))

#train_ds[1] = 0.8
#train_ds[1,:,:,3] = 1

uncompressed_image = Image.fromarray((train_ds[1]*255).astype(np.uint8),"RGBA")
uncompressed_image.save('uncompress_image.png',format='png')

compressor_2(train_ds[1], compressed_image, metadata('degree'), metadata('number_of_theta_of_same_X'), epochs=metadata('epoch'))
"""
for count, image in enumerate(decompressed_image):
  decompressed_image_to_save = Image.fromarray((image*255).astype(np.uint8),"RGBA")
  path_progress_images = "Decompress_images_progress/"+str(count+1)+".png"
  decompressed_image_to_save.save(path_progress_images,format='png')
"""

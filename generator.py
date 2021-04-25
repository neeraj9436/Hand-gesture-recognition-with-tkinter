import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import load_model
import operator
import cv2
import math
import sys, os
from tkinter import *
from tkinter import Message ,Text
import tkinter.ttk as ttk
import tkinter.font as font
from PIL import ImageTk,Image
import time
print("Imported")



##################################################################################################################
def collect_sample():
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/train")
##        os.makedirs("data1/test")
        os.makedirs("data/train/0")
        os.makedirs("data/train/1")
        os.makedirs("data/train/2")
        os.makedirs("data/train/3")
        os.makedirs("data/train/4")
        os.makedirs("data/train/5")
        os.makedirs("data/train/6")
        os.makedirs("data/train/left")
        os.makedirs("data/train/right")
        os.makedirs("data/train/ok")


    mode = 'train'
    directory = 'data/'+mode+'/'

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        
        # Getting count of existing images
        count = {'zero': len(os.listdir(directory+"/0")),
                 'one': len(os.listdir(directory+"/1")),
                 'two': len(os.listdir(directory+"/2")),
                 'three': len(os.listdir(directory+"/3")),
                 'four': len(os.listdir(directory+"/4")),
                 'five': len(os.listdir(directory+"/5")),
                 'six': len(os.listdir(directory+"/6")),
                 'left': len(os.listdir(directory+"/left")),
                 'right': len(os.listdir(directory+"/right")),
                 'ok': len(os.listdir(directory+"/ok"))}


        cv2.putText(frame, "PUT HAND IN BOX AND CLICK No. ", (5, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.putText(frame, "SIX : "+str(count['six']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.putText(frame, "LEFT : "+str(count['left']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.putText(frame, "RIGHT : "+str(count['right']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.putText(frame, "OK : "+str(count['ok']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)


        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (150,150)) 
     
        cv2.imshow("Frame", frame)

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 140, 255, cv2.THRESH_BINARY)
        cv2.imshow("ROI", roi)

        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        if interrupt & 0xFF == ord('0'):
            cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', roi)
        if interrupt & 0xFF == ord('1'):
            cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', roi)
        if interrupt & 0xFF == ord('2'):
            cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', roi)
        if interrupt & 0xFF == ord('3'):
            cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', roi)
        if interrupt & 0xFF == ord('4'):
            cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', roi)
        if interrupt & 0xFF == ord('5'):
            cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', roi)
        if interrupt & 0xFF == ord('6'):
            cv2.imwrite(directory+'6/'+str(count['six'])+'.jpg', roi)
        if interrupt & 0xFF == ord('L'):
            cv2.imwrite(directory+'left/'+str(count['left'])+'.jpg', roi)
        if interrupt & 0xFF == ord('R'):
            cv2.imwrite(directory+'right/'+str(count['right'])+'.jpg', roi)
        if interrupt & 0xFF == ord('O'):
            cv2.imwrite(directory+'ok/'+str(count['ok'])+'.jpg', roi)

    cap.release()
    cv2.destroyAllWindows()

#collect_sample()



#######################################################################################################################

def training_sample():
    classifier = Sequential()

    classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=6, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Code copied from - https://keras.io/preprocessing/image/


    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory('data/train',
                                                     target_size=(64, 64),
                                                     batch_size=5,
                                                     color_mode='grayscale',
                                                     class_mode='categorical')

    test_set = test_datagen.flow_from_directory('data/test',
                                                target_size=(64, 64),
                                                batch_size=5,
                                                color_mode='grayscale',
                                                class_mode='categorical')




    classifier.fit(
            training_set,
            steps_per_epoch=240, # No of images in training set
            epochs=1,
            validation_data=test_set,
            validation_steps=30)# No of images in test set


    # Saving the model
    model_json = classifier.to_json()
    with open("model-bw.json", "w") as json_file:
        json_file.write(model_json)
    classifier.save_weights('model-bw.h5')



    print("Model training complete")
#######################################################################################################

def predict():

    loaded_model = tf.keras.models.load_model('myt2.h5')
    loaded_model.summary()
##    print("Loaded model from disk")

    cap = cv2.VideoCapture(0)

    # Category dictionary
    
    PWD = "12345"
    new_pwd = ""
    dx = 5
    dy = 5
    interval = 8
    b = 100
    r = 0
    g = 200
    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)

        #time.sleep(2)
        # Got this from collect-data.py
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (150,150)) 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        test2 = np.array(test_image,dtype='float32')
        test2 = test2/255
        #test2 = test2/255
        test2 = test2.astype(np.float64)
        #print(test2)
        #test2 = (test2-0.0032502487)/(0.0014753705)
        cv2.imshow("test", test2)
       

        
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == ord('5'):
                result = loaded_model.predict(test2.reshape(1, 150,150, 1)) 
                #print(result[0])
                #print(test2.reshape(1, 150,150, 1))
                
                pas_key = {'0': result[0][0], 
                           '1': result[0][1], 
                           '2': result[0][2],
                           '3': result[0][3],
                           '4': result[0][4],
                           '5': result[0][5],
                           '6': result[0][6], 
                           'L': result[0][7], 
                           'R': result[0][9],
                           'O': result[0][8]}

                prediction = {'Zero': result[0][0], 
                              'One': result[0][1], 
                              'Two': result[0][2],
                              'Three': result[0][3],
                              'Four': result[0][4],
                              'Five': result[0][5],
                              'Six': result[0][6], 
                              'Left': result[0][7], 
                              'Right': result[0][9],
                              'Ok': result[0][8]
                              }
                # Sorting based on top prediction
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                #print(prediction)
                pas_key = sorted(pas_key.items(), key=operator.itemgetter(1), reverse=True)
                new_pwd = new_pwd + pas_key[0][0]
                # Displaying the predictions
                cv2.putText(frame, str(prediction[0][0]), (10, 150), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 3)
        if interrupt & 0xFF == ord('C'):
             new_pwd = ""
        if new_pwd == PWD:
            cv2.putText(frame, "!! VALIDATED !!", (130,50), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0),3)
            cv2.putText(frame, "!! WELCOME USER !!", (-100+dx, 200), cv2.FONT_HERSHEY_PLAIN, 3, (b,g,r),3)
            cv2.putText(frame, "!! WELCOME USER !!", (600-dx, 250), cv2.FONT_HERSHEY_PLAIN, 3, (g,r,b),3)
            cv2.putText(frame, "!! WELCOME USER !!", (-300+2*dx, 300), cv2.FONT_HERSHEY_PLAIN, 3, (r,b,g),3)
            if dx >= 800:
                interval = interval * -1
            elif dx <=4:
                interval = interval * -1
            dx = dx + interval
            b=(b+1)%255
            r=(r+1)%255
            g=(g+1)%255
            #print(dx)
            
        else:
            cv2.putText(frame, "Press 'C' to clear pass:", (0, 400), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
            cv2.putText(frame, "Password: " + new_pwd, (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3) 
        cv2.imshow("Frame", frame)
        
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
            
     
    cap.release()
    cv2.destroyAllWindows()



##collect_sample()
##training_sample()
##predict()



###################################################################################################################################################################
def convexhull():
    cap = cv2.VideoCapture(0)
    def nothing(x):
        pass
    cv2.namedWindow('Trackbar')
    cv2.createTrackbar('h1','Trackbar',0,255,nothing)
    cv2.createTrackbar('s1','Trackbar',0,255,nothing)
    cv2.createTrackbar('v1','Trackbar',10,255,nothing)
    cv2.createTrackbar('h2','Trackbar',255,255,nothing)
    cv2.createTrackbar('s2','Trackbar',180,255,nothing)
    cv2.createTrackbar('v2','Trackbar',144,255,nothing)    
    while(1):
            
        try:  #an error comes if it does not find anything in window as it cannot find contour of max area
              #therefore this try error statement
              
            ret, frame = cap.read()
            frame=cv2.flip(frame,1)
            kernel = np.ones((3,3),np.uint8)
            
            #define region of interest
            roi=frame[100:300, 100:300]
            
            
            cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            
             
        # define range of skin color in HSV
    ##    lower_skin = np.array([2,0,0], dtype=np.uint8)
    ##    upper_skin = np.array([20,255,255], dtype=np.uint8)
            h1 = cv2.getTrackbarPos('h1','Trackbar')
            s1 = cv2.getTrackbarPos('s1','Trackbar')
            v1 = cv2.getTrackbarPos('v1','Trackbar')
            h2 = cv2.getTrackbarPos('h2','Trackbar')
            s2 = cv2.getTrackbarPos('s2','Trackbar')
            v2 = cv2.getTrackbarPos('v2','Trackbar')
                
           ## mask2 = cv2.inRange(hsv,np.array([2,0,0]),np.array([20,255,255]))
            mask = cv2.inRange(hsv,np.array([h1,s1,v1]),np.array([h2,s2,v2]))
         #extract skin colur imagw  
            #mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
       
            
        #extrapolate the hand to fill dark spots within
##            mask = cv2.erode(mask,kernal,iterations=1)
##            mask = cv2.dilate(mask,kernel,iterations = 4)
            
        #blur the image
            mask = cv2.GaussianBlur(mask,(5,5),100) 
            
            
            
        #find contours
            _,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
       #find contour of max area(hand)
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
            
        #approx the contour a little
            epsilon = 0.0005*cv2.arcLength(cnt,True)
            approx= cv2.approxPolyDP(cnt,epsilon,True)
           
            
        #make convex hull around hand
            hull = cv2.convexHull(cnt)
            
         #define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
          
        #find the percentage of area not covered by hand in convex hull
            arearatio=((areahull-areacnt)/areacnt)*100
        
         #find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
            
        # l = no. of defects
            l=0
            
        #code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt= (100,180)
                
                
                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                
                #distance between point and convex hull
                d=(2*ar)/a
                
                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                
            
                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d>30:
                    l += 1
                    cv2.circle(roi, far, 3, [255,0,0], -1)
                
                #draw lines around hand
                cv2.line(roi,start, end, [0,255,0], 2)
                
                
            l+=1
            
            #print corresponding gestures which are in their ranges
            font = cv2.FONT_HERSHEY_SIMPLEX
            if l==1:
                if areacnt<2000:
                    cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    if arearatio<12:
                        cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    elif arearatio<17.5:
                        cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                       
                    else:
                        cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        
            elif l==2:
                cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                
            elif l==3:
             
                  if arearatio<27:
                        cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                  else:
                        cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        
            elif l==4:
                cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                
            elif l==5:
                cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                
            elif l==6:
                cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                
            else :
                cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                
            #show the windows
            cv2.imshow('mask',mask)
            cv2.imshow('frame',frame)
        except:
            pass
            
        
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        
    cv2.destroyAllWindows()
    cap.release()    
#########################################################################################################################
                                        ##Final##




win = Tk()
win.geometry("1400x700")
img1 = Image.open(r"C:\Users\user\Desktop\Programming\hand gesture\number-sign-recognition-master\nature1.jpg")
resized = img1.resize((1400,700),Image.ANTIALIAS)
canvas = Canvas(win,width=1400,height=700)
image = ImageTk.PhotoImage(resized)
#image = image.resize((1000,600),Image.ANTIALIAS)
canvas.create_image(0,0,anchor=NW,image=image)
canvas.pack()

win.title("Hand Gesture Recognition (Exp. project)")
#win.configure(background='blue')

message = Label(win, text="Hand-Gesture-Recognition System" ,bg="Green"  ,fg="white"  ,width=30  ,height=2,font=('times', 30, 'italic bold underline'))
message.place(x=300,y=0)

lbl = Label(win, text="Want to collect Sample : ",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
lbl.place(x=300, y=200)

train_button = Button(win,text="Click to collect",width=10,height=1,fg='black',font=('times', 15, ' bold '),command=collect_sample)
train_button.place(x=650,y=205)

lb2 = Label(win, text="Train the Model : ",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
lb2.place(x=300, y=300)

train_button = Button(win,text="Click to Train",width=10,height=1,fg='black',font=('times', 15, ' bold '),command=training_sample)
train_button.place(x=650,y=305)

lb3 = Label(win, text="Make Prediction : ",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
lb3.place(x=300, y=400)

train_button = Button(win,text="Click to Predict",width=13,height=1,fg='black',font=('times', 15, ' bold '),command=predict)
train_button.place(x=650,y=405)

lb4 = Label(win, text="Prediction with Convexhull : ",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
lb4.place(x=300, y=500)

con_button = Button(win,text="Click to Predict",width=13,height=1,fg='black',font=('times', 15, ' bold '),command=convexhull)
con_button.place(x=650,y=505)

message = Label(win, text="Thanks" ,bg="yellow"  ,fg="black"  ,font=('times', 15, 'italic bold underline'))
message.place(x=1000,y=600)
message = Label(win, text="Prof M.K. Mesharam" ,bg="Green"  ,fg="white"  ,font=('times', 20, 'italic bold underline'))
message.place(x=950,y=650)

message = Label(win, text="Team Member:" ,bg="yellow"  ,fg="red"  ,font=('times', 20, 'italic bold underline'))
message.place(x=1100,y=250)
message = Label(win, text="Neeraj Kumar Yadav" ,bg="blue"  ,fg="white"  ,font=('times', 15,'bold underline'))
message.place(x=1150,y=300)
message = Label(win, text="Niket Singh" ,bg="blue"  ,fg="white"  ,font=('times', 15, 'bold underline'))
message.place(x=1150,y=350)
message = Label(win, text="Nisitha" ,bg="blue"  ,fg="white"  ,font=('times', 15, 'bold underline'))
message.place(x=1150,y=400)
message = Label(win, text="Nitesh Gupta" ,bg="blue"  ,fg="white"  ,font=('times', 15, 'bold underline'))
message.place(x=1150,y=450)

win.mainloop()























    








    

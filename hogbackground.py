import cv2 
import numpy as np
from math import atan2,degrees,sqrt
import matplotlib.pyplot as plt
#now after opening the opencv2
#we try to read the image
#we will implement this with the class
class Image_processing:
    def __init__(self,image_path):
        self.image_path=image_path
        #now we would read the image from the image path 
        self.image=cv2.imread(self.image_path)
        #now we convert the image to gray image
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        
        print(self.image_path)
        print(self.image.shape)
        
    def show_image(self):
        cv2.imshow("Image",self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def resize_image(self):
        #here we want to resize the image to bring it in the form of 128x64 
        smaller_image=cv2.resize(self.image,(128,64),interpolation=cv2.INTER_AREA)
        #this will reduce the area
        self.image=smaller_image #we updated our image 
        #now we print the shape of the new image
        
        self.length,self.breadth=self.image.shape
        #we have defined the breadth and the length
        print(self.image.shape)
        
    def calculate_gradients_x(self):
        self.gx=[]  #universal magnitude and the angle 
        
        for i in range(self.length):
            for j in range(self.breadth):
                if(j-1<0 or j+1>=64):
                    if(j-1<0):
                        Gx=(self.image[i][j+1]).astype(np.int32)
                    elif(j+1>=64):
                        Gx=-1*self.image[i][j-1].astype(np.int32)
                else:
                    Gx=self.image[i][j+1].astype(np.int32)-self.image[i][j-1].astype(np.int32)
                self.gx.append(Gx) 
        #now lets change the dimension of the array
        self.gx=np.asarray(self.gx)
        self.gx=self.gx.reshape(self.length,self.breadth)
        
    
    def calculate_gradients_y(self):
        self.gy=[]
        
        for i in range(self.length):
            for j in range(self.breadth):
                if(i-1<0  or i+1>=64):
                    if (i-1<0):
                        Gy=self.image[i+1][j].astype(np.int32)
                    elif(i+1>=64):
                        Gy=(-1)*self.image[i-1][j].astype(np.int32)
                else:
                    Gy=self.image[i+1][j].astype(np.int32)-self.image[i-1][j].astype(np.int32)
                self.gy.append(Gy)
    
                
        #now lets change the dimension of the array
        self.gy=np.asarray(self.gy)
        self.gy=self.gy.reshape(self.length,self.breadth)
        
    def show_gradients(self):
        print(self.gx)
        print("------------------------------")
        print(self.gy)
        print("-------------------------------")
        
        print(" Shape is ",self.gx.shape)
        print("Shape is  ",self.gy.shape)
        
    def compute_gradients(self):
        self.magnitude=[]
        self.orientation=[]
        for i in range(self.length):
            mags=[]
            orients=[]
            for j in range(self.breadth):
                mag=sqrt(self.gx[i][j]**2+self.gy[i][j]**2)
                orient=round(degrees(atan2(self.gy[i][j],self.gx[i][j])))
                mags.append(mag)
                orients.append(orient)
                
            self.magnitude.append(mags)
            self.orientation.append(orients)
        #now we resize the matrix
        
        self.magnitude=np.asarray(self.magnitude)
        
        
        self.orientation=np.asarray(self.orientation)
        self.orientation=self.orientation.reshape(self.length,self.breadth)
           
        print(self.magnitude)
        print(self.orientation)            
        
    def visualization_magnitude_of_image(self):
        plt.figure(figsize=(10,10))
        plt.imshow(self.magnitude,cmap='gray',vmin=0,vmax=255)
        plt.axis("off")
        plt.show()
        
    def visualization_orientation_of_image(self):
        plt.figure(figsize=(10,10))
        plt.imshow(self.orientation,cmap='gray',vmin=0,vmax=255)
        plt.axis("off")
        plt.show()
        
    
    def divide_into_blocks(self):
        self.final_orientation=[]
        num_rows, num_cols = self.orientation.shape
        
        num_blocks_rows = num_rows // 8
        num_blocks_cols = num_cols // 8
        blocks_matrix = np.empty((num_blocks_rows, num_blocks_cols))
        rows=0
        for r in range(0, num_rows, 8):
            rows=rows+1
            columns=0
            for c in range(0, num_cols, 8):
                columns=columns+1
                block = self.orientation[r:r+8, c:c+8]
                block=block.reshape(-1,64)
                self.final_orientation.append(block)
        print(rows)
        print(columns)
        self.final_orientation=np.asarray(self.final_orientation)
        self.final_orientation=self.final_orientation.reshape(128,64)
        print("this si the self.final_orientation")
        print(self.final_orientation.shape)
        
        
    def orientation_in_each_block(self):
        #we divide into 9 bins for the orientation
        #0,20,40,60,80,100,120,140,160
        #so we define a count variable for each part 
        self.histogram_for_each=[]
        self.orientations=[]
        
        
        
        for i in range(len(self.final_orientation)):
            l=self.final_orientation[i]
            count=[0]*9
            for j in range(len(l)):
                
                if l[j]>=160:
                    count[8]=count[8]+1
                elif 160>l[j]>=140 :
                    count[7]=count[7]+1
                elif 140>l[j]>=120:
                    count[6]=count[6]+1
                elif 120>l[j]>=100:
                    count[5]=count[5]+1
                elif 100>l[j]>=80:
                    count[4]=count[4]+1
                elif 80>l[j]>=60:
                    count[3]=count[3]+1
                elif 60>l[j]>=40:
                    count[2]=count[2]+1
                elif 40>l[j]>=20:
                    count[1]=count[1]+1
                else :
                    count[0]=count[0]+1
        #now we want to find the orientation of the block 
        #so for that we see the max of the above
            self.histogram_for_each.append(count)
            self.orientations.append(np.argmax(count)*20)
            
            #change them into array
        self.histogram_for_each=np.asarray(self.histogram_for_each)
        self.orientations=np.asarray(self.orientations)
            
            #change them into corresponding dimensions
            
        self.orientations=self.orientations.reshape(16,8)
            
        print(self.orientations)
            
            
        #this will give histogram values for all the values
        #this is not the normalized hog
        
        
#lets create a object of the class
img=Image_processing("/Users/arpitsingh/Desktop/opencv projects/opencv-4.7.0/samples/data/ellipses.jpg")


img.resize_image()
img.calculate_gradients_x()
img.calculate_gradients_y()
img.show_gradients()
img.compute_gradients()

#here we visualize our magnitude and the orientation
img.visualization_magnitude_of_image()
img.visualization_orientation_of_image()


img.divide_into_blocks()
img.orientation_in_each_block()
        

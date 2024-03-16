import cv2
import numpy as np
#we will implement the edge detectors classes , here we will use 5 types of edge detectors
#the parameters of cv2.Filter2D is 
#source image , destination image , depth of the image , concolutional kernel
class Edge_detectors:
    def __init__(self,path_name):
        self.pathname=path_name
        self.image=cv2.imread(self.pathname) #here we store the pathname 
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY) #this changes the color to grayscale
        self.image_blur=cv2.GaussianBlur(self.image,(3,3),0)
        
    #now we use the different edge detector methods
    #roberts,sobel,laplace,canny,scharr,prewitt
    def sobel_filter(self):
        
        #now after blurring the image ,we need to apply the sobel filter
        sobelx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobely=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        edges_x=cv2.filter2D(self.image_blur,cv2.CV_32F,sobelx)
        edges_y=cv2.filter2D(self.image_blur,cv2.CV_32F,sobely)

        edges_combined=np.sqrt(edges_x**2+edges_y**2)

#scale the values between 0-255
        edges_combined = cv2.normalize(edges_combined, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8
        edges_combined = np.uint8(edges_combined) 
        
        self.sobel_image=edges_combined
        
        
    def prewitt_filter(self):
        
        prewittx=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        prewitty=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        edges_x=cv2.filter2D(self.image_blur,cv2.CV_32F,prewittx)
        edges_y=cv2.filter2D(self.image_blur,cv2.CV_32F,prewitty)

        edges_combined=np.sqrt(edges_x**2+edges_y**2)

#scale the values between 0-255
        edges_combined = cv2.normalize(edges_combined, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8
        edges_combined = np.uint8(edges_combined) 
        
        self.prewitt_image=edges_combined
        
        
    def roberts_filter(self):
        
        robertx=np.array([[1,0],[0,-1]])
        roberty=np.array([[0,1],[-1,0]])
        
        edges_x=cv2.filter2D(self.image_blur,cv2.CV_32F,robertx)
        edges_y=cv2.filter2D(self.image_blur,cv2.CV_32F,roberty)

        edges_combined=np.sqrt(edges_x**2+edges_y**2)

#scale the values between 0-255
        edges_combined = cv2.normalize(edges_combined, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8
        edges_combined = np.uint8(edges_combined) 
        
        self.roberts_image=edges_combined
        
    def scharr_filter(self):
        
        scharrx=np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
        scharry=np.array([[3,10,3],[0,0,0],[-3,10,-3]])
        
        edges_x=cv2.filter2D(self.image_blur,cv2.CV_32F,scharrx)
        edges_y=cv2.filter2D(self.image_blur,cv2.CV_32F,scharry)

        edges_combined=np.sqrt(edges_x**2+edges_y**2)

#scale the values between 0-255
        edges_combined = cv2.normalize(edges_combined, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8
        edges_combined = np.uint8(edges_combined) 
        
        self.scharr_image=edges_combined
        
        
    def laplacian_filter(self):
        
        laplacian_filter=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        
        edges=cv2.filter2D(self.image_blur,cv2.CV_32F,laplacian_filter)
        
        self.laplacian_image=edges
        
    def canny_filter(self):
        
        self.canny_image=cv2.Canny(self.image_blur,50,150)
        
    def operations(self):
        self.sobel_filter()
        self.prewitt_filter()
        self.roberts_filter()
        self.laplacian_filter()
        self.scharr_filter()
        self.canny_filter()
        
    def showcase_the_edges_images(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 3
        font_color = (255, 0, 0)  # White color

        # Add titles to each image
        cv2.putText(self.image,"original image", (20, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(self.image_blur,"blurred image", (20, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(self.sobel_image, "sobel filter", (20, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(self.prewitt_image, "prewitt filter", (20, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(self.roberts_image, "roberts filter", (20, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(self.scharr_image, "scharr filter", (20, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(self.laplacian_image, "laplacian filter", (20, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(self.canny_image, "canny filter", (20, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        #result=np.hstack((self.image,self.image_blur,self.sobel_image,self.prewitt_image,self.roberts_image,self.scharr_image,self.laplacian_image,self.canny_image))
        imga=cv2.imread("/Users/arpitsingh/Desktop/opencv projects/opencv-4.7.0/samples/data/ela_modified.jpg")
        cv2.imshow("",self.image)
        cv2.waitKey(0)
        cv2.imshow("",self.image_blur)
        cv2.waitKey(0)
        cv2.imshow("",self.sobel_image)
        cv2.waitKey(0)
        cv2.imshow("",self.prewitt_image)
        cv2.waitKey(0)
        cv2.imshow("",self.roberts_image)
        cv2.waitKey(0)
        cv2.imshow("",self.laplacian_image)
        cv2.waitKey(0)
        cv2.imshow("",self.canny_image)
        cv2.waitKey(0)
        cv2.imshow("",self.scharr_image)
        cv2.waitKey(0)
        
        #cv2.imshow("RESULTS",result)
   
        cv2.destroyAllWindows()
        #now we print the images 


#here we create an obUsers/arpitsingh/Desktop/opencv projects/opencv-4.7.0/samples/data/ela_modified.jpgject of the class     
img=Edge_detectors("/Users/arpitsingh/Desktop/opencv projects/opencv-4.7.0/samples/data/ela_modified.jpg")

img.operations()
img.showcase_the_edges_images()



        
        
        
        
        
        

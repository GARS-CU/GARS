import pandas
import math

class Focus_Calibrator:
    def __init__(self, openface_csv_path: str) -> None:

        

        gaze_angle_x, gaze_angle_y = self.openface_to_numpy(openface_csv_path)

        self.gaze_angle_xmin = gaze_angle_x.min()
        self.gaze_angle_xmax = gaze_angle_x.max()
        self.gaze_angle_xcenter = (self.gaze_angle_xmax + self.gaze_angle_xmin)/2
        
        self.gaze_angle_ymin = gaze_angle_x.min()
        self.gaze_angle_ymax = gaze_angle_y.max()
        self.gaze_angle_ycenter = (self.gaze_angle_ymax + self.gaze_angle_ymin)/2
  
      
    
    def openface_to_numpy(self, openface_csv_path: str):
        openface_csv_df = pandas.read_csv(openface_csv_path)
        openface_csv_df = openface_csv_df[[' gaze_angle_x', ' gaze_angle_y']]

        openface_numpy = openface_csv_df.to_numpy()

        gaze_angle_x = openface_numpy[:,0]
        gaze_angle_y = openface_numpy[:,1]

        return gaze_angle_x, gaze_angle_y
        
    def __call__(self, gaze_angle_x, gaze_angle_y) -> int:
        """function used to determine focus on a linear scale.
        determines fraction of distance from center of focus to outside focus
        on a linear scale."""
        distance = lambda x, y : math.sqrt((x)**2 + (y)**2)
        #find distance from center
        x_distance = (gaze_angle_x-self.gaze_angle_xcenter) 
        y_distance = (gaze_angle_y-self.gaze_angle_ycenter)
        distance_from_center_to_gaze = distance(x_distance, y_distance)
        

        radians = math.atan2(y_distance, x_distance)
        slope = y_distance/x_distance
        
        b = self.gaze_angle_ycenter-slope*self.gaze_angle_xcenter

        radians = math.atan2(y_distance, x_distance)
    
        if(radians <= math.pi/4 and radians > -math.pi/4):
            #right wall
            y = slope*self.gaze_angle_xmax+b
            distance_from_center = distance(self.gaze_angle_xmax-self.gaze_angle_xcenter, y - self.gaze_angle_ycenter)
    
        elif(radians > math.pi/4 and radians <= (3*math.pi/4)):
            #upper wall
            x = (self.gaze_angle_ymax - b)/slope
            distance_from_center = distance(x-self.gaze_angle_xcenter, self.gaze_angle_ymax - self.gaze_angle_ycenter)

        elif(radians > (3*math.pi/4) and radians >= -(3*math.pi/4)):
            #left wall
            y = slope*self.gaze_angle_xmin+b
            distance_from_center = distance(self.gaze_angle_xmin - self.gaze_angle_xcenter, y - self.gaze_angle_ycenter)
        else:
            #bottom wall
            x = (self.gaze_angle_ymax - b)/slope
            distance_from_center = distance(x - self.gaze_angle_xcenter, self.gaze_angle_ymin - self.gaze_angle_ycenter)

        eye_gaze_focus = 1 - (distance_from_center_to_gaze/distance_from_center)      
        
        return eye_gaze_focus
    

    #test
        
print(Focus_Calibrator('~/processed/demo.csv')(-0.2, 0.06))
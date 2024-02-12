import pandas
import math
import numpy as np 

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
        distance = lambda x, y : np.sqrt((x)**2 + (y)**2)
        #find distance from center
        x_distance = (gaze_angle_x-self.gaze_angle_xcenter) 
        y_distance = (gaze_angle_y-self.gaze_angle_ycenter)
        distance_from_center_to_gaze = distance(x_distance, y_distance)
        

        
        slope = y_distance/x_distance
        
        b = self.gaze_angle_ycenter-slope*self.gaze_angle_xcenter

        radians = np.arctan2(y_distance, x_distance)
        distance_from_center = np.zeros_like(distance_from_center_to_gaze)
      
    
        #right wall
        right_mask = (radians <= math.pi/4) and (radians > -math.pi/4)

        y_right_wall = slope[right_mask]*self.gaze_angle_xmax+b[right_mask]
        distance_from_center[right_mask] = distance(self.gaze_angle_xmax-self.gaze_angle_xcenter, y_right_wall - self.gaze_angle_ycenter)
        
        #upper wall
        upper_mask = (radians > math.pi/4) and (radians <= 3*math.pi/4)
        x_upper_wall = (self.gaze_angle_ymax - b[upper_mask])/slope[upper_mask]
        distance_from_center[upper_mask] = distance(x_upper_wall -self.gaze_angle_xcenter, self.gaze_angle_ymax - self.gaze_angle_ycenter)
        
        
        #left wall
        left_mask = (radians > (3*np.pi/4) and radians >= -3*np.pi/4)
        y_left_wall = slope[left_mask]*self.gaze_angle_xmin+b
        distance_from_center[left_mask] = distance(self.gaze_angle_xmin - self.gaze_angle_xcenter, y_left_wall - self.gaze_angle_ycenter)
    

        #bottom wall
        bottom_mask = radians < -np.pi/4 and radians >= -(3*np.pi/4)
        y_bottom_wall = (self.gaze_angle_ymax - b[bottom_mask])/slope[bottom_mask]
  
        distance_from_center[bottom_mask] = distance( y_bottom_wall  - self.gaze_angle_xcenter, self.gaze_angle_ymin - self.gaze_angle_ycenter)
    
        eye_gaze_focus = 1 - (distance_from_center_to_gaze/distance_from_center)    

        return eye_gaze_focus
    


        
print(Focus_Calibrator('~/processed/demo.csv')(np.array([-0.2]), np.array([0.06]))) 
import pandas
class Focus_Calibrator:
    def __init__(self, openface_csv_path: str) -> None:
        openface_csv_df = pandas.read_csv(openface_csv_path)

        openface_csv_df = openface_csv_df[[' gaze_angle_x', ' gaze_angle_y']]

        openface_numpy = openface_csv_df.to_numpy()

        gaze_angle_x = openface_numpy[:,0]
        gaze_angle_y = openface_numpy[:,1]

        self.gaze_angle_xmin = gaze_angle_x.min()
        self.gaze_angle_xmax = gaze_angle_x.max()
        self.gaze_angle_xcenter = (self.gaze_angle_xmax + self.gaze_angle_xmin)/2
        
        self.gaze_angle_ymin = gaze_angle_x.min()
        self.gaze_angle_ymax = gaze_angle_y.max()
        self.gaze_angle_ycenter = (self.gaze_angle_ymax + self.gaze_angle_ymin)/2
      

        
    def __call__(self, gaze_angle_x, gaze_angle_y) -> int:

        return 2
    

    #test
        
Focus_Calibrator('~/processed/demo.csv')
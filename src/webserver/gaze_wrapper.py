import os
import sys

sys.path.append(os.path.join(os.environ['GARS_PROJ'], 'engagement', 'focus'))
from focus_calibrator import Focus_Calibrator
class GazeWrapper:
    """
    wrapper for gaze model
    """
    def __init__(self) -> None:
        self._gaze_model  = None

    def start(self, init_csv):
        self._gaze_model = Focus_Calibrator(init_csv)

    def __call__(self, filename):
        if self._gaze_model == None:
            return None
        cur_csv = os.path.join(f"/zooper2/{os.getenv('USER')}/openface_dump/processed", f"{os.path.splitext(filename)[0]}.csv")
        return self._gaze_model(cur_csv)



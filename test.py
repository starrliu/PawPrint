from pawprint.event.chasing import ChasingDetector
from pawprint.data import TrajectoryCollection
import json

output_file = "data/chasing_events.json"

trajectory_collection = TrajectoryCollection("data/gt.csv", fps=30, scale=0.259)

chasing_detector = ChasingDetector(trajectory_collection, 
                                   theta_threshold=0.5,
                                   phi_threshold=-0.1, 
                                   correlation_threshold=0.3, 
                                   min_movement=20, 
                                   distance_threshold=30,
                                   window_size=120)

events = chasing_detector.detect()

# Save events to json
result = [event.to_dict() for event in events]

with open(output_file, "w") as f:
    json.dump(result, f)
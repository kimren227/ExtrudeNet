import json
import os

class Config:

    def __init__(self, config_path):
        with open(config_path) as json_file:
            config_dict = json.load(json_file)
        self.experiment_name = os.path.splitext(os.path.basename(config_path))[0]

        # Hardware related
        self.num_gpu = config_dict["num_gpu"]
        self.train_batch_size_per_gpu = config_dict["train_batch_size_per_gpu"]
        self.test_batch_size_per_gpu = config_dict["test_batch_size_per_gpu"]

        # CSG-Stump
        self.num_primitives = config_dict["num_primitives"]
        self.num_bezier_segments = config_dict["num_bezier_segments"]
        self.num_intersections = config_dict["num_intersections"]
        self.feature_dim = config_dict["feature_dim"]
        self.extrude_dir = config_dict["extrude_dir"]
        self.sharpness = config_dict["sharpness"]
        self.connection_mode = config_dict["connection_mode"]
        if "extrude_shape" not in config_dict:
            self.extrude_shape = "bezier"
        else:
            self.extrude_shape = config_dict["extrude_shape"]

        if "continuity" not in config_dict:
            self.continuity = "C1"
        else:
            self.continuity = config_dict["continuity"]


        if "sample_rate" not in config_dict:
            self.sample_rate = 100
        else:
            self.sample_rate = config_dict["sample_rate"]

        if "use_polar_theta" not in config_dict:
            self.use_polar_theta = False
        else:
            self.use_polar_theta = config_dict["use_polar_theta"]

        # Optimizer
        self.learning_rate = config_dict["learning_rate"]
        self.beta1 = config_dict["beta1"]

        # Training
        self.epoch = config_dict["epoch"]
        self.eval_interval = config_dict["eval_interval"]

        # Eval
        self.real_size = config_dict["real_size"]
        self.test_size = config_dict["test_size"]
        self.csg_dir = config_dict["csg_dir"]
        self.sample_dir = config_dict["sample_dir"]

        # Dataset
        self.dataset_root = config_dict["dataset_root"]

        self.num_surface_points = config_dict["num_surface_points"]
        self.num_sample_points = config_dict["num_sample_points"]
        self.category = config_dict["category"]
        self.balance = config_dict["balance"]

        # Loss
        self.scale_primitive_loss = config_dict["scale_primitive_loss"]
        self.scale_drift_loss = config_dict["scale_drift_loss"]
        self.scale_polygon_loss = config_dict["scale_control_polygon_loss"]
        self.scale_support_loss = config_dict["scale_support_loss"]




if __name__ == "__main__":
    conf = Config("./configs/plane_256_64.json")









import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from sdfs import *
from dgcnn import DGCNNFeat

class CSGStump(nn.Module):
    def __init__(self, num_primitives, num_intersections, num_bezier_segments, sharpness, sample_rate):
        super(CSGStump, self).__init__()
        self.num_primitives = num_primitives
        self.num_intersections = num_intersections
        self.num_bezier_segments = num_bezier_segments
        self.sharpness = sharpness
        self.sample_rate = sample_rate

        # Compute polar angles for each control point, please refer to Sec.3.1 of the paper for details
        radians = []
        theta = (math.pi * 2) / (4 * self.num_bezier_segments) + math.atan((1 / 3) * math.tan(2 * math.pi / (4 * self.num_bezier_segments)))
        for i in range(self.num_bezier_segments):
            radians.append(2 * math.pi / self.num_bezier_segments * i)
            radians.append(2 * math.pi / self.num_bezier_segments * i + theta)
            radians.append(2 * math.pi / self.num_bezier_segments * (i + 1) - theta)

        control_polygon_radians = torch.tensor(np.asarray(radians).astype(np.float32))
        control_polygon_radians = control_polygon_radians.view(self.num_bezier_segments, 3).cuda() # [P,4]
        self.control_polygon_radians = control_polygon_radians.unsqueeze(0).unsqueeze(0) # [1,1,P,3]


    def forward(self, sample_point_coordinates, primitive_parameters, intersection_layer_weights, union_layer_weights, is_training):

        primitive_parameters = primitive_parameters.transpose(2,1)
        B, K, _ = primitive_parameters.shape
        quaternion = primitive_parameters[:,:,:4]
        translation = primitive_parameters[:,:,4:7]
        height = primitive_parameters[:,:,7:8].squeeze(-1)

        control_polygon_radius = torch.abs(primitive_parameters[:, :, 8:8+self.num_bezier_segments*3].view(B, K, self.num_bezier_segments, 3))
        control_polygon_radians = self.control_polygon_radians.repeat(B,K,1,1) # [B, K, P, 3]

        # control_pillar_radius [BKP2]
        # weights [B, K, P, 2]
        # interpolate last control_radius
        control_radians_x = torch.cos(control_polygon_radians) * control_polygon_radius # [B, K, P, 3]
        control_radians_y = torch.sin(control_polygon_radians) * control_polygon_radius # [B, K, P, 3]

        control_polygon = torch.stack([control_radians_x, control_radians_y], dim=-1) # [B, K, P, 3, 2]

        weights = torch.abs(primitive_parameters[:,:,8+self.num_bezier_segments*3:].view(B,K,self.num_bezier_segments,2))

        primitive_sdf, support_distances = sdf_extruded_bezier(quaternion, translation, sample_point_coordinates, control_polygon, weights,  height, self.sample_rate, sdf_mode="Barycentric")

        primitive_occupancies = torch.sigmoid(-1 * primitive_sdf * self.sharpness)


        # calculate intersections
        # W * occupancy + (1-W) * 1,  where 1 indicates solid, i.e. solid intersect anything is equal to itself
        occupancy_pre_intersection = torch.einsum("bkc,bmk->bmkc", intersection_layer_weights, primitive_occupancies) \
                                                    + torch.einsum("bkc,bmk->bmkc", 1-intersection_layer_weights, primitive_occupancies.new_ones(primitive_occupancies.shape))
        if not is_training:
            intersection_node_occupancies = torch.min(occupancy_pre_intersection, dim=-2)[0]
        else:
            with torch.no_grad():
                # use soft min to distribute gradients
                weights = torch.softmax(occupancy_pre_intersection * (-40), dim=-2)
            intersection_node_occupancies = torch.sum(weights * occupancy_pre_intersection, dim=-2) # [BMC]

        # calculate union
        # W*sdf + (1-W)*(0) where 0 indicates empty, and empty union anything is equal to itself
        occupancy_pre_union = torch.einsum("bc,bmc->bmc", union_layer_weights, intersection_node_occupancies)
        if not is_training:
            occupancies = torch.max(occupancy_pre_union, dim=-1)[0]
        else:
            with torch.no_grad():
                # use soft max to distribute gradients
                weights = torch.softmax(occupancy_pre_union * (40), dim=-1)
            occupancies = torch.sum(weights  * occupancy_pre_union, dim=-1)
        return occupancies, primitive_sdf, intersection_node_occupancies, support_distances


class CSGStumpConnectionHead(nn.Module):
    def __init__(self, feature_dim, num_primitives, num_intersections):
        super(CSGStumpConnectionHead, self).__init__()
        self.num_primitives = num_primitives
        self.num_intersections = num_intersections
        self.feature_dim = feature_dim
        self.intersection_linear = nn.Linear(self.feature_dim * 8, self.num_primitives * self.num_intersections, bias=True)
        self.union_linear = nn.Linear(self.feature_dim * 8, self.num_intersections, bias=True)

    def forward(self, feature, is_training):
        # getting intersection layer connection weights
        intersection_layer_weights = self.intersection_linear(feature)
        intersection_layer_weights = intersection_layer_weights.view(-1, self.num_primitives, self.num_intersections) # [B, num_primitives, num_intersections]

        # getting union layer connection weights
        union_layer_weights = self.union_linear(feature)
        union_layer_weights = union_layer_weights.view(-1, self.num_intersections) # [B,c_dim]

        if not is_training:
            # during inference, we use descrtize connection weights to get interpretiable CSG relations
            intersection_layer_weights = (intersection_layer_weights>0).type(torch.float32)
            union_layer_weights = (union_layer_weights>0).type(torch.float32)
        else:
            # during train, we use continues connection weights to get better gradients
            intersection_layer_weights = torch.sigmoid(intersection_layer_weights)
            union_layer_weights = torch.sigmoid(union_layer_weights)

        return intersection_layer_weights, union_layer_weights

class CSGStumpConnectionFixedHead(nn.Module):
    def __init__(self, feature_dim, num_primitives, num_intersections):
        super(CSGStumpConnectionFixedHead, self).__init__()
        self.num_primitives = num_primitives
        self.num_intersections = num_intersections
        intersection_layer_weights = torch.zeros((self.num_primitives, self.num_intersections))
        union_layer_weights = torch.zeros((self.num_intersections))
        self.intersection_layer_weights = nn.Parameter(intersection_layer_weights)
        self.union_layer_weights = nn.Parameter(union_layer_weights)
        nn.init.normal_(self.intersection_layer_weights)
        nn.init.normal_(self.union_layer_weights)

    def forward(self, feature, is_training):
        if not is_training:
            # during inference, we use descrtize connection weights to get interpretiable CSG relations
            intersection_layer_weights = (self.intersection_layer_weights>0).type(torch.float32)
            union_layer_weights = (self.union_layer_weights>0).type(torch.float32)
        else:
            # during train, we use continues connection weights to get better gradients
            intersection_layer_weights = torch.sigmoid(self.intersection_layer_weights)
            union_layer_weights = torch.sigmoid(self.union_layer_weights)

        return intersection_layer_weights.unsqueeze(0).repeat(feature.shape[0], 1, 1), union_layer_weights.unsqueeze(0).repeat(feature.shape[0], 1)


class CSGStumpConnectionUnionHead(nn.Module):
    def __init__(self, feature_dim, num_primitives, num_intersections):
        super(CSGStumpConnectionUnionHead, self).__init__()
        assert (num_primitives == num_intersections), "union only requires intersection nodes and number of primitive to be equal"
        self.num_primitives = num_primitives
        self.num_intersections = num_intersections
        self.intersection_layer_weights = torch.eye(self.num_primitives).cuda()
        self.union_layer_weights = torch.ones(self.num_intersections).cuda()

    def forward(self, feature, is_training):
        return self.intersection_layer_weights.unsqueeze(0).repeat(feature.shape[0], 1, 1), self.union_layer_weights.unsqueeze(0).repeat(feature.shape[0], 1)

class CSGStumpPrimitiveHead(nn.Module):

    def __init__(self, feature_dim, num_primitives, num_bezier_segments, extrude_dir):
        super(CSGStumpPrimitiveHead, self).__init__()
        self.num_primitives = num_primitives
        self.feature_dim = feature_dim
        # You may contraint the extrusion direction if needed
        self.extrude_dir = extrude_dir
        assert extrude_dir in ["free", "ortho", "side", "top", "front"], "extrude_dir must be one of free, ortho, side, top, front"
        if self.extrude_dir != "free":
            self.num_primitive_parameters = 3 + 1 + num_bezier_segments * 3 + num_bezier_segments*2
        else:
            self.num_primitive_parameters = 3 + 4 + 1+num_bezier_segments * 3 + num_bezier_segments*2
        self.num_type = 4
        self.primitive_linear = nn.Linear(self.feature_dim * 8, self.num_primitives * self.num_primitive_parameters, bias=True)
        nn.init.xavier_uniform_(self.primitive_linear.weight)
        nn.init.constant_(self.primitive_linear.bias, 0)

    def forward(self, feature):
        shapes = self.primitive_linear(feature)
        shapes = shapes.view(-1, self.num_primitive_parameters, self.num_primitives)
        if self.extrude_dir == "ortho":
            num_extrusion_each_direction =  int(self.num_primitives/4)
            # rotate to front view
            identity_quaternion_front = shapes.new_zeros([shapes.shape[0], 4, num_extrusion_each_direction])
            identity_quaternion_front[:,0,:] = 1
            # rotate to top view
            identity_quaternion_top = shapes.new_zeros([shapes.shape[0], 4, num_extrusion_each_direction])
            identity_quaternion_top[:,0,:] = 0.707
            identity_quaternion_top[:,1,:] = 0.707
            # rotate to side view
            identity_quaternion_side = shapes.new_zeros([shapes.shape[0], 4, num_extrusion_each_direction*2])
            identity_quaternion_side[:,0,:] = 0.707
            identity_quaternion_side[:,-2,:] = 0.707
            identity_quaternion = torch.cat([identity_quaternion_front, identity_quaternion_top, identity_quaternion_side], dim=-1)
            shapes = torch.cat([identity_quaternion, shapes], dim=1)
            return shapes
        elif self.extrude_dir == "side":
            # rotate to side view
            identity_quaternion_side = shapes.new_zeros([shapes.shape[0], 4, self.num_primitives])
            identity_quaternion_side[:,0,:] = 0.707
            identity_quaternion_side[:,-2,:] = 0.707
            shapes = torch.cat([identity_quaternion_side, shapes], dim=1)
        elif self.extrude_dir == "top":
            # rotate to top view
            identity_quaternion_top = shapes.new_zeros([shapes.shape[0], 4, self.num_primitives])
            identity_quaternion_top[:,0,:] = 0.707
            identity_quaternion_top[:,1,:] = 0.707
            shapes = torch.cat([identity_quaternion_top, shapes], dim=1)
        elif self.extrude_dir == "front":
            # rotate to front view
            identity_quaternion_front = shapes.new_zeros([shapes.shape[0], 4, self.num_primitives])
            identity_quaternion_front[:,0,:] = 1
            shapes = torch.cat([identity_quaternion_front, shapes], dim=1)
        else:
            return shapes

        return shapes

class Decoder(nn.Module):

    def __init__(self, feature_dim):
        super(Decoder, self).__init__()
        self.feature_dim = feature_dim
        self.linear_1 = nn.Linear(self.feature_dim, self.feature_dim * 2, bias=True)
        self.linear_2 = nn.Linear(self.feature_dim * 2, self.feature_dim * 4, bias=True)
        self.linear_3 = nn.Linear(self.feature_dim * 4, self.feature_dim * 8, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)

    def forward(self, inputs):
        l1 = self.linear_1(inputs)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)
        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)
        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)
        return l3

class ExtrudeNet(nn.Module):

    def __init__(self, config):
        super(ExtrudeNet, self).__init__()
        self.config = config

        self.num_primitives = self.config.num_primitives
        self.num_intersections = self.config.num_intersections
        self.num_bezier_segments = self.config.num_bezier_segments
        self.feature_dim = self.config.feature_dim
        self.sharpness = self.config.sharpness
        self.extrude_dir = self.config.extrude_dir
        self.connection_mode = self.config.connection_mode
        self.sample_rate = self.config.sample_rate
        self.use_polar_theta = self.config.use_polar_theta
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder(self.feature_dim)
        self.connection_head = CSGStumpConnectionHead(self.feature_dim, self.num_primitives, self.num_intersections)
        self.primitive_head = CSGStumpPrimitiveHead(self.feature_dim, self.num_primitives, self.num_bezier_segments, self.extrude_dir)
        self.csg_stump = CSGStump(self.num_primitives, self.num_intersections, self.num_bezier_segments, self.sharpness, self.sample_rate)

    def forward(self, surface_pointcloud, sample_coordinates, is_training=True):
        feature = self.encoder(surface_pointcloud)
        code = self.decoder(feature)
        intersection_layer_connections, union_layer_connections = self.connection_head(code, is_training=is_training)
        primitive_parameters = self.primitive_head(code)
        occupancies, primitive_sdfs, _, support_distances = self.csg_stump(sample_coordinates, primitive_parameters, intersection_layer_connections, union_layer_connections, is_training=is_training)
        return occupancies, primitive_sdfs, primitive_parameters, support_distances

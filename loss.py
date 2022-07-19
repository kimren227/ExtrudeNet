import torch
import torch.nn as nn
import math

class PrimitiveLoss(nn.Module):

    def __init__(self, config):
        super(PrimitiveLoss, self).__init__()
        self.scale = config.scale_primitive_loss

    def forward(self, primitive_sdf):
        primitive_loss = torch.mean((primitive_sdf.min(dim=1)[0])**2) * self.scale
        return primitive_loss

class ReconLoss(nn.Module):

    def __init__(self, config):
        super(ReconLoss, self).__init__()

    def forward(self, pred_point_value, gt_point_value):
        loss_recon = 4 * torch.mean((pred_point_value - gt_point_value)**2)
        return loss_recon

class WeightsLoss(nn.Module):

    def __init__(self, config):
        super(WeightsLoss, self).__init__()

    def forward(self, primitive_parameters):
        weights = torch.abs(primitive_parameters[:,:,16:])
        loss_recon = 10 * torch.mean((weights-1)**2)
        return loss_recon

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.extrude_shape = config.extrude_shape
        self.primitive_loss = PrimitiveLoss(config)
        self.recon_loss = ReconLoss(config)
        self.weight_loss = WeightsLoss(config)

    def forward(self, predict_occupancy, gt_occupancy, primitive_sdf, primitive_parameters, support_distances):
        loss_recon = self.recon_loss(predict_occupancy, gt_occupancy)
        loss_primitive = self.primitive_loss(primitive_sdf)
        loss_weights = self.weight_loss(primitive_parameters)
        loss_total = loss_recon + loss_primitive + loss_weights
        return {"loss_recon":loss_recon, "loss_weights": loss_weights, "loss_primitive":loss_primitive, "loss_total":loss_total}









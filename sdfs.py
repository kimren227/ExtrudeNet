import torch
import torch.nn as nn
import chamfer2D.dist_chamfer_2D
import math
chamfer2d = chamfer2D.dist_chamfer_2D.chamfer_2DDist()
# quaternion code are copied from pytorch3d
def standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a, b):
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def quaternion_invert(quaternion):
    return quaternion * quaternion.new_tensor([1, -1, -1, -1])

def quaternion_apply(quaternion, point):
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

def transform_points(quaternion, translation, points):
    quaternion = nn.functional.normalize(quaternion, dim=-1)
    transformed_points = points.unsqueeze(2) - translation.unsqueeze(1)
    transformed_points = quaternion_apply(quaternion.unsqueeze(1), transformed_points)
    return transformed_points

def convert_duplet_to_control_points(control_pillar):
    # control_pillar [B,K,P,2,2]
    # control_points [B,K,P,4,2]
    P = control_pillar.shape[-3]
    control_points = []
    
    # interpolate the shared control points
    # first control point is midpoint of last point of the previous control duplet and the first point of current control duplet
    # last control point is midpoint of last point of the current control duplet and the first point of next control duplet

    control_points.append(torch.stack([(control_pillar[..., -1, :, :][..., 1, :] + control_pillar[..., 0, :, :][..., 0, :]) / 2, 
                                    control_pillar[..., 0, :, :][..., 0, :],
                                    control_pillar[..., 0, :, :][..., 1, :],
                                    (control_pillar[..., 0, :, :][..., 1, :] + control_pillar[..., 1, :, :][..., 0, :]) / 2], dim=-2))

    for i in range(1, P - 1):
        control_points.append(torch.stack([(control_pillar[..., i-1, :, :][..., 1, :] + control_pillar[..., i, :, :][..., 0, :]) / 2, 
                                    control_pillar[..., i, :, :][..., 0, :],
                                    control_pillar[..., i, :, :][..., 1, :],
                                    (control_pillar[..., i, :, :][..., 1, :] + control_pillar[..., i+1, :, :][..., 0, :]) / 2], dim=-2))
    # close the loop
    control_points.append(torch.stack([(control_pillar[..., -2, :, :][..., 1, :] + control_pillar[..., -1, :, :][..., 0, :]) / 2, 
                                    control_pillar[..., -1, :, :][..., 0, :],
                                    control_pillar[..., -1, :, :][..., 1, :],
                                    (control_pillar[..., -1, :, :][..., 1, :] + control_pillar[..., 0, :, :][..., 0, :]) / 2], dim=-2))

    return torch.stack(control_points, dim=-3)

def convert_triplet_to_control_points(control_triplet):
    # control_triplet [B,K,P,3,2]
    # control_points [B,K,P,4,2]
    B,K,P,_,_ = control_triplet.shape
    shared_points = control_triplet[:,:,:,0:1,:]
    shared_points = shared_points.roll(-1,2)
    control_points = torch.cat([control_triplet, shared_points], dim=-2)
    return control_points

def sample_closed_bezier_curves(control_points, t, return_normal):
    # control_points [B, K, P, 4, 2]
    # t [S]
    B, K, P, _, _ = control_points.shape
    S = t.shape[0]
    control_points = control_points.unsqueeze(-3).repeat(1,1,1,S,1,1)
    t = t.view(1,1,1,S,1)

    p = control_points[..., 0, :] * ((1-t)**3) + \
        control_points[..., 1, :] * 3 * ((1-t) ** 2) * t + \
        control_points[..., 2, :] * 3 * (1-t) * (t ** 2) + \
        control_points[..., 3, :] * (t ** 3)
    first_sample = p[:,:,:,0:1,:]
    last_sample = p.roll(1,2)[:,:,:,-1:,:]
    # average_sample = (first_sample + last_sample)/2
    # p = torch.cat([p[:,:,:,1:-1,:], average_sample], dim=-2)

    if return_normal:
        dp = 3 * ((control_points[..., 3, :] - 3* control_points[..., 2, :] + 3 * control_points[..., 1, :] - control_points[..., 0, :]) * t**2 + \
                (2*control_points[..., 2, :] - 4 * control_points[..., 1, :] + 2 * control_points[..., 0, :]) * t + \
                    control_points[..., 1, :] - control_points[..., 0, :])

        n = torch.stack([-dp[...,-1], dp[...,0]], dim=-1)
        n = torch.nn.functional.normalize(n, dim=-1)
        # n [B,K,P,S,2]
        first_normal = n[:,:,:,0:1,:]
        last_normal = n.roll(1,2)[:,:,:,-1:,:]
        # average_normal = (first_normal + last_normal)/2
        # n = torch.cat([n[:,:,:,1:-1,:], average_normal], dim=-2)
        # return p.view(B, K, P*(S-1), 2), n.view(B, K, P*(S-1), 2)
        return p.view(B, K, P*S, 2), n.view(B, K, P*S, 2)

    else:
        # return p.view(B, K, P*(S-1), 2)
        return p.view(B, K, P*S, 2)

def sample_closed_rational_bezier_curves(control_points, t, weights, return_normal):
    # control_points [B, K, P, 4, 2]
    # weights [B, K, P, 2]
    # t [S]
    B, K, P, _, _ = control_points.shape
    S = t.shape[0]
    control_points = control_points.unsqueeze(-3).repeat(1,1,1,S,1,1)
    t = t.view(1,1,1,S,1) # [1,1,1,S,1]
    
    # Bernstein polynomial
    B0 = (1-t)**3
    B1 = 3 * ((1-t) ** 2) * t
    B2 = 3 * (1-t) * (t ** 2)
    B3 = (t ** 3)

    # First order derivative of basis functions
    dB0 = -3*t**2 + 6*t - 3
    dB1 = 9*t**2 - 12*t + 3
    dB2 = -9*t**2 + 6*t
    dB3 = 3*t**2 

    # getting control points and weights
    P0 = control_points[..., 0, :]
    P1 = control_points[..., 1, :]
    P2 = control_points[..., 2, :]
    P3 = control_points[..., 3, :]

    W1 = weights[...,0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,S,2) # [BKP] -> [BKPS2]
    W2 = weights[...,1].unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,S,2) # [BKP] -> [BKPS2]

    # compute nominator and denominator
    gt = P0*B0 + W1*P1*B1 + W2*P2*B2 + P3*B3
    ht = B0 + W1*B1 + W2*B2 + B3

    # final sample points
    p =  gt/ht

    # first_sample = p[:,:,:,0:1,:]
    # last_sample = p.roll(1,2)[:,:,:,-1:,:]
    # average_sample = (first_sample + last_sample)/2

    # p = torch.cat([p[:,:,:,1:-1,:], average_sample], dim=-2)


    if return_normal:
        # compute unormalized tangent vectors
        dgdt = P0*dB0 + W1*P1*dB1 + W1*P2*dB2 + P3*dB3 
        dhdt = dB0 + W1*dB1 + W2*dB2 + dB3
        dp = (dgdt*ht + dhdt*gt)/(ht**2)

        n = torch.stack([-dp[...,-1], dp[...,0]], dim=-1)
        n = torch.nn.functional.normalize(n, dim=-1)
        # n [B,K,P,S,2]
        # first_normal = n[:,:,:,0:1,:]
        # last_normal = n.roll(1,2)[:,:,:,-1:,:]
        # average_normal = (first_normal + last_normal)/2
        # n = torch.cat([n[:,:,:,1:-1,:], average_normal], dim=-2)
        # return p.view(B, K, P*(S-1), 2), n.view(B, K, P*(S-1), 2)
        return p.view(B, K, P*S, 2), n.view(B, K, P*S, 2)

    else:
        return p.view(B, K, P*S, 2)

def sample_curve(control_polygon, weights, num_points_per_segment, return_normal):
    B,K,P,D,_ = control_polygon.shape
    t = torch.arange(0, 1, 1/num_points_per_segment).cuda()
    if D == 2:
        control_points = convert_duplet_to_control_points(control_polygon)
    elif D==3:
        control_points = convert_triplet_to_control_points(control_polygon)
    else:
        raise ValueError('Control Polygon must be 2 or 3')
    if weights is not None:
        return sample_closed_rational_bezier_curves(control_points, t, weights, return_normal)
    else:
        return sample_closed_bezier_curves(control_points, t, return_normal)

def sdf_2d_bezier(control_polygon, weights, testing_points, sample_rate, sdf_mode='Barycentric'):
    assert sdf_mode in ['General', 'Barycentric'], "sdf_mode must be either 'General' or 'Barycentric'"
    # Batch size, Number of Sketch, Number of Curve per Sketch, Number of control points, Number of dimensions
    B,K,P,D,_ = control_polygon.shape
    if sdf_mode == "General":
        sample_points, sample_normals = sample_curve(control_polygon, weights, sample_rate, True)
        # reshape the tensor for parallel distance computation
        sample_normals = sample_normals.reshape(B*K,sample_normals.shape[-2], sample_normals.shape[-1])

    else:
        sample_points = sample_curve(control_polygon, weights, sample_rate, False)
    
    # PS: total number of points sampled from the sketch
    # M: total number of testing points
    _, _, PS , _ = sample_points.shape
    _, M, _, _ = testing_points.shape


    # reshape the tensor for parallel distance computation
    testing_points = testing_points.transpose(2,1)
    testing_points = testing_points.reshape(B*K,testing_points.shape[-2], testing_points.shape[-1])
    sample_points = sample_points.reshape(B*K,sample_points.shape[-2], sample_points.shape[-1])
    distances, support_distances, indice, _ = chamfer2d(testing_points[..., :2], # [B*K,M,2]
                                                    sample_points # [B*K,S,2]
                                                    )
    # Distance computed is squared distance, so we need to take the square root
    distances = torch.sqrt(distances) # [B*K,M]
    indice = indice.unsqueeze(-1).type(torch.int64)
    # We do not need to compute the sign's gradient
    with torch.no_grad():
        if sdf_mode == "General":
            nearest_normal = torch.gather(sample_normals, 1, indice.expand(indice.shape[0], indice.shape[1], sample_normals.shape[-1]))
            nearest_normal = nn.functional.normalize(nearest_normal, dim=-1)
            testing_normal = testing_points[..., :2] - torch.gather(sample_points, 1, indice.expand(indice.shape[0], indice.shape[1], sample_normals.shape[-1]))
            cos = (nearest_normal * testing_normal).sum(-1)
            sign = -cos/(torch.abs(cos)+1e-9)
        else:
            # Compute the sign based on barycentric coordinates

            # Find which quadrant the point is in
            testing_points_in_orthant_1 = torch.bitwise_and(testing_points[..., 0] >= 0, testing_points[..., 1] >= 0)
            testing_points_in_orthant_23 = testing_points[..., 0] < 0
            testing_points_in_orthant_4 = torch.bitwise_and(testing_points[..., 0] >= 0, testing_points[..., 1] < 0)

            sample_points_in_orthant_1 = torch.bitwise_and(sample_points[..., 0] >= 0, sample_points[..., 1] >= 0)
            sample_points_in_orthant_23 = sample_points[..., 0] < 0
            sample_points_in_orthant_4 = torch.bitwise_and(sample_points[..., 0] >= 0, sample_points[..., 1] < 0)

            # Compute the point in the polar coordinate system
            # Quadrant:
            #   I	Use the arctan value
            #   II	Add 180° to the arctan value
            #   III	Add 180° to the arctan value
            #   IV	Add 360° to the arctan value

            delta_testing_points_radian = torch.stack([testing_points_in_orthant_1.type(torch.float32) * 0,
                                                    testing_points_in_orthant_23.type(torch.float32) * math.pi, 
                                                    testing_points_in_orthant_4.type(torch.float32)* math.pi*2], dim=-1).sum(dim=-1)
            testing_points_radian = torch.arctan(testing_points[..., 1]/testing_points[...,0]) + delta_testing_points_radian # [B*K, M]

            delta_sample_points_radian = torch.stack([sample_points_in_orthant_1.type(torch.float32) * 0,
                                                    sample_points_in_orthant_23.type(torch.float32) * math.pi, 
                                                    sample_points_in_orthant_4.type(torch.float32)* math.pi*2], dim=-1).sum(dim=-1)

            sample_points_radian = torch.arctan(sample_points[..., 1]/sample_points[...,0]) + delta_sample_points_radian # [B*K, S]

            # Find the nearest sample point in the polar coordinate system
            difference = torch.abs(testing_points_radian.unsqueeze(1) - sample_points_radian.unsqueeze(-1)) # [B*K, S, M]
            polar_indice = difference.min(dim=1)[1]



            # As the point could fall in the previous triangle or the next triangle, we need to find both of them
            indice_next = ((polar_indice + 1) % (PS)).unsqueeze(-1)
            indice_pre = ((polar_indice - 1) % (PS)).unsqueeze(-1)
            indice_nearest = ((polar_indice) % (PS)).unsqueeze(-1)

            nearest_next_points = torch.gather(sample_points, 1, indice_next.expand(indice.shape[0], indice.shape[1], sample_points.shape[-1])) # [B*K, M, 2]
            nearest_pre_points = torch.gather(sample_points, 1, indice_pre.expand(indice.shape[0], indice.shape[1], sample_points.shape[-1]))# [B*K, M, 2]
            nearest_points = torch.gather(sample_points, 1, indice_nearest.expand(indice.shape[0], indice.shape[1], sample_points.shape[-1]))# [B*K, M, 2]

            pre_triangles = torch.stack([nearest_pre_points.view(B*K*M, -1), nearest_points.view(B*K*M, -1)], dim=-2)
            next_triangles = torch.stack([nearest_points.view(B*K*M, -1), nearest_next_points.view(B*K*M, -1)], dim=-2)

            # Compute the barycentric coordinates in previous and next triangle
            pre_bx = pre_triangles[:,0,0]# [B*K*M]
            pre_by = pre_triangles[:,0,1]# [B*K*M]
            pre_cx = pre_triangles[:,1,0]# [B*K*M]
            pre_cy = pre_triangles[:,1,1]# [B*K*M]

            next_bx = next_triangles[:,0,0]# [B*K*M]
            next_by = next_triangles[:,0,1]# [B*K*M]
            next_cx = next_triangles[:,1,0]# [B*K*M]
            next_cy = next_triangles[:,1,1]# [B*K*M]

            points = testing_points[...,:2].view(B*K*M,2)
            # Compute braycentric coordinates for previous triangle
            pre_betas = (-pre_cy * points[:,0] + pre_cx * points[:,1])/(-pre_cy * pre_bx + pre_cx * pre_by)
            pre_gammas = (-pre_by * points[:,0] + pre_bx * points[:,1])/(-pre_by * pre_cx + pre_bx * pre_cy)
            pre_alphas = 1-pre_betas-pre_gammas
            # Compute if the point is in the previous triangle
            inside_pre_triangle = torch.stack([torch.abs(pre_alphas-0.5)<=0.5, torch.abs(pre_betas-0.5)<=0.5, torch.abs(pre_gammas-0.5)<=0.5], dim=-1).sum(dim=-1)==3

            # compute braycentric coordinates for next triangle
            next_betas = (-next_cy * points[:,0] + next_cx*points[:,1])/(-next_cy*next_bx + next_cx*next_by)
            next_gammas = (-next_by * points[:,0] + next_bx*points[:,1])/(-next_by*next_cx + next_bx*next_cy)
            next_alphas = 1-next_betas-next_gammas
            # Compute if the point is in the next triangle
            inside_next_triangle = torch.stack([torch.abs(next_alphas-0.5)<=0.5, torch.abs(next_betas-0.5)<=0.5, torch.abs(next_gammas-0.5)<=0.5], dim=-1).sum(dim=-1)==3
            sign = ((inside_pre_triangle + inside_next_triangle).type(torch.float32)-0.5).sign() * -1
            sign = sign.view(B,K,M).view(B*K, M)

    sdfs = (sign * distances).squeeze(-1)
    indice = indice.squeeze(-1)
    support_distances = support_distances.squeeze(-1)
    # shape them back
    sdfs = sdfs.reshape(B, K, sdfs.shape[-1]).transpose(2,1)

    sample_points = sample_points.reshape(B, K, sample_points.shape[-2], sample_points.shape[-1])
    support_distances = support_distances.reshape(B, K, support_distances.shape[-1]).transpose(2,1)
    indice = indice.reshape(B, K, indice.shape[-1]).transpose(2,1)
    return sdfs, sample_points, support_distances

def sdf_extruded_bezier(quaternion, translation, points, control_polygon, weights, extrude_height, sample_rate, sdf_mode="Barycentric"):
    extrude_height = torch.abs(extrude_height)
    transformed_points = transform_points(quaternion, translation, points)
    sdf_2d, sample_points, support_distances = sdf_2d_bezier(control_polygon, weights, transformed_points, sample_rate, sdf_mode)
    h = torch.abs(transformed_points[..., -1]) - extrude_height.unsqueeze(-2)
    d = torch.stack([sdf_2d, h], dim=-1)
    return d[...,0].max(d[...,1]).clamp_max(0) + torch.norm(d.clamp_min(0), dim=-1), support_distances

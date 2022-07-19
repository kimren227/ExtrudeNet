import os
import time
import shutil

import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import ShapeNet
from model import ExtrudeNet
from loss import Loss
from config import Config
from utils import init
import argparse

device = torch.device("cuda")

def train(config):

    init(config)

    train_loader= DataLoader(ShapeNet(shapenet_root=config.dataset_root, balance=True, categories=[config.category,], partition="train"), pin_memory=True, num_workers=40, batch_size=config.train_batch_size_per_gpu*config.num_gpu, shuffle=True, drop_last=True)
    test_loader = DataLoader(ShapeNet(shapenet_root=config.dataset_root, balance=True, categories=[config.category,], partition="val"), pin_memory=True, num_workers=40, batch_size=config.test_batch_size_per_gpu*config.num_gpu, shuffle=True, drop_last=True)

    # clear pervious tensorboard entries
    if os.path.exists("./runs/%s" % config.experiment_name):
        shutil.rmtree("./runs/%s" % config.experiment_name)
    writer = SummaryWriter("./runs/%s" % config.experiment_name)

    # loading model
    model = ExtrudeNet(config).to(device)
    pre_train_model_path = './checkpoints/%s/models/model.th' % config.experiment_name

    if not os.path.exists(pre_train_model_path):
        print("Cannot find pre-train model for experiment: {}\n Training from scratch".format(config.experiment_name))
    else:
        print("Loading pre-train weights from {}".format(pre_train_model_path))
        model.load_state_dict(torch.load('./checkpoints/%s/models/model.th' % config.experiment_name))

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    # training settings
    opt=torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    criterion = Loss(config)
    start_time = time.time()
    eval_interval = config.eval_interval
    train_counter = 0
    test_counter = 0
    current_loss_recon = np.inf

    for epoch in range(config.epoch):
        model.train()
        avg_loss = avg_loss_recon = avg_loss_primitive = avg_loss_weights = avg_loss_drift = avg_loss_support = avg_loss_control_polygon = iter_counter = avg_accuracy = avg_recall = avg_fscore = avg_fscore = 0
        print("Training: Epoch %d" % epoch)
        train_loader_t = tqdm(train_loader)
        for surface_pointcloud, testing_points in train_loader_t:
            surface_pointcloud = surface_pointcloud.to(device)
            testing_points = testing_points.to(device)
            model.zero_grad()
            occupancies, primitive_sdfs, primitive_parameters, support_distances = model(surface_pointcloud.transpose(2,1), testing_points[:,:,:3], is_training=True)
            predict_occupancies = (occupancies >=0.5).float()
            target_occupancies = (testing_points[:,:,-1] >=0.5).float()
            accuracy = torch.sum(predict_occupancies*target_occupancies)/torch.sum(target_occupancies)
            recall = torch.sum(predict_occupancies*target_occupancies)/(torch.sum(predict_occupancies)+1e-9)

            loss_dict = criterion(occupancies, testing_points[:,:,-1], primitive_sdfs, primitive_parameters, support_distances)
            loss_dict["loss_total"].backward()
            train_loader_t.set_description("Loss Total: %f" % loss_dict["loss_total"].item())
            opt.step()

            # accumulate loss values
            avg_loss += loss_dict["loss_total"].item()
            avg_loss_recon += loss_dict["loss_recon"].item()
            avg_loss_weights += loss_dict["loss_weights"].item()

            avg_loss_primitive += loss_dict["loss_primitive"].item()
            avg_loss_drift += loss_dict["loss_drift"].item()
            avg_loss_support += loss_dict["loss_support"].item()

            avg_accuracy += accuracy.item()
            avg_recall += recall.item()
            fscore = 2*accuracy.item()*recall.item()/(accuracy.item() + recall.item() + 1e-6)
            fscore = 0 if np.isnan(fscore) else fscore
            avg_fscore += fscore

            # update tensorboard
            writer.add_scalar('Loss/train',loss_dict["loss_total"].item(), global_step=train_counter)
            writer.add_scalar("Loss_Recon/train", loss_dict["loss_recon"].item(),global_step=train_counter)
            writer.add_scalar("Loss_Primitive/train", loss_dict["loss_primitive"].item(),global_step=train_counter)
            writer.add_scalar("Loss_Drift/train", loss_dict["loss_drift"].item(),global_step=train_counter)
            writer.add_scalar("Loss_Support/train", loss_dict["loss_support"].item(),global_step=train_counter)
            writer.add_scalar("Loss_Control_Polygon/train", loss_dict["loss_control_polygon"].item(),global_step=train_counter)


            writer.add_scalar("accuracy/train", accuracy.item(), global_step=train_counter)
            writer.add_scalar("recall/train", recall.item(), global_step=train_counter)
            writer.add_scalar("fscore/train", fscore, global_step=train_counter)
            iter_counter += 1
            train_counter += 1

        print("Experiment: %s" % config.experiment_name)
        print("Training: [%2d/%2d] time: %4.4f, loss_total: %.6f, loss_recon: %.6f, loss_drift: %6f, loss_support: %6f, loss_control_polygon: %6f, loss_primitive: %.6f, loss_weights: %.6f, acc: %.6f, recall: %.6f, fscore: %.6f" % (epoch,
                                                                                            config.epoch,
                                                                                            time.time() - start_time,
                                                                                            avg_loss/iter_counter,
                                                                                            avg_loss_recon / iter_counter,
                                                                                            avg_loss_drift / iter_counter,
                                                                                            avg_loss_support / iter_counter,
                                                                                            avg_loss_control_polygon / iter_counter,
                                                                                            avg_loss_primitive/iter_counter,
                                                                                            avg_loss_weights/iter_counter,
                                                                                            avg_accuracy  / iter_counter,
                                                                                            avg_recall  / iter_counter,
                                                                                            avg_fscore  / iter_counter))
        # Eval models
        if (epoch+1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                testloader_t = tqdm(test_loader)
                avg_test_loss_recon = avg_test_loss_primitive = avg_test_loss_drift = avg_test_loss_support = avg_test_loss_control_polygon = avg_test_loss = test_iter_counter = avg_test_accuracy = avg_test_fscore= avg_test_recall = 0
                for surface_pointcloud, testing_points  in testloader_t:

                    surface_pointcloud = surface_pointcloud.to(device)
                    testing_points = testing_points.to(device)

                    occupancies, primitive_sdfs, primitive_parameters, support_distances = model(surface_pointcloud.transpose(2,1), testing_points[:,:,:3], is_training=False)
                    loss_dict = criterion(occupancies, testing_points[:,:,-1], primitive_sdfs, primitive_parameters, support_distances)

                    predict_occupancies = (occupancies >=0.5).float()
                    target_occupancies = (testing_points[:,:,-1] >=0.5).float()

                    accuracy = torch.sum(predict_occupancies*target_occupancies)/torch.sum(target_occupancies)
                    recall = torch.sum(predict_occupancies*target_occupancies)/(torch.sum(predict_occupancies)+1e-9)

                    avg_test_loss_recon += loss_dict["loss_recon"].item()
                    avg_test_loss_primitive += loss_dict["loss_primitive"].item()
                    avg_test_loss_drift += loss_dict["loss_drift"].item()
                    avg_test_loss_support += loss_dict["loss_support"].item()
                    avg_test_loss_control_polygon += loss_dict["loss_control_polygon"].item()
                    avg_test_loss += loss_dict["loss_total"].item()
                    avg_test_accuracy += accuracy.item()
                    avg_test_recall += recall.item()
                    test_fsocre = 2*accuracy.item()*recall.item()/(accuracy.item() + recall.item() + 1e-6)
                    avg_test_fscore += test_fsocre

                    # update tensorboard
                    writer.add_scalar('Loss/test', loss_dict["loss_total"].item(), global_step=test_counter)
                    writer.add_scalar("Loss_Primitive/test", loss_dict["loss_primitive"].item(),global_step=test_counter)
                    writer.add_scalar("Loss_Recon/test", loss_dict["loss_recon"].item(),global_step=test_counter)
                    writer.add_scalar("Loss_Support/test", loss_dict["loss_support"].item(),global_step=test_counter)
                    writer.add_scalar("Loss_Control_Polygon/test", loss_dict["loss_control_polygon"].item(),global_step=test_counter)



                    writer.add_scalar("accuracy/test", accuracy.cpu().detach().numpy(),global_step=test_counter)
                    writer.add_scalar("recall/test", recall.cpu().detach().numpy(),global_step=test_counter)
                    writer.add_scalar("fscore/test", test_fsocre, global_step=test_counter)

                    test_counter += 1
                    test_iter_counter += 1

                print("Testing: [%2d/%2d] time: %4.4f, loss_total: %.6f, loss_recon: %.6f, loss_drift: %.6f, loss_support: %.6f, loss_control_polygon: %.6f, loss_primitive: %.6f, acc: %.6f, recall: %.6f, fscore: %.6f" % (epoch,
                                                                                                            config.epoch,
                                                                                                            time.time() - start_time,
                                                                                                            avg_test_loss/test_iter_counter,
                                                                                                            avg_test_loss_recon / test_iter_counter,
                                                                                                            avg_test_loss_drift / test_iter_counter,
                                                                                                            avg_test_loss_support / test_iter_counter,
                                                                                                            avg_test_loss_control_polygon / test_iter_counter,
                                                                                                            avg_test_loss_primitive/test_iter_counter,
                                                                                                            avg_test_accuracy / test_iter_counter,
                                                                                                            avg_test_recall / test_iter_counter,
                                                                                                            avg_test_fscore / test_iter_counter))
                if avg_test_loss_recon < current_loss_recon:
                    current_loss_recon = avg_test_loss_recon
                    print("Updating model weights... Current best epoch: %d" % (epoch+1))
                    torch.save(model.module.state_dict(), './checkpoints/%s/models/model.th' % config.experiment_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ExtrudeNet')
    parser.add_argument('--config_path', type=str, default='./configs/plane.json', metavar='N',
                        help='config_path')
    args = parser.parse_args()
    config = Config((args.config_path))
    train(config)






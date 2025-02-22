from __future__ import print_function
import sys
import os
os.chdir('KANbeFair/src')
# It is recommended to use an absolute path here
import argparse
import warnings
import torch
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
from fvcore.common.timer import Timer

from utils import *
from models.kan.LBFGS import *
warnings.simplefilter(action='ignore', category=UserWarning)

def train(args, model, device, train_loader, optimizer, epoch, logger, start_index):
    model.train()
    if epoch == 1 and start_index == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {args.model}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        for name, param in model.named_parameters():
            logger.info(f"Layer: {name} | Parameters: {param.numel()} | Trainable: {param.requires_grad}")

    for batch_idx, (data, target) in enumerate(train_loader, start_index):
        data, target = todevice(data, device), todevice(target, device)

        if args.optimizer in ["adam",'sgd']:
            
            optimizer.zero_grad()
            output = model(data)

            if args.loss == "cross_entropy":
                losses = [F.cross_entropy(output, target)]
            elif args.loss == "mse":
                losses = [F.mse_loss(output, target)]
            else:
                raise NotImplementedError
            
            loss = 0
            for l in losses:
                loss = loss + l
            loss.backward()
            optimizer.step()

        elif args.optimizer == "lbfgs":
            # print("lbfgs")

            def closure():
                optimizer.zero_grad()
                output = model(data)
                if args.loss == "cross_entropy":
                    losses = [F.cross_entropy(output, target)]
                elif args.loss == "mse":
                    losses = [F.mse_loss(output, target)]
                else:
                    raise NotImplementedError

                loss = 0
                for l in losses:
                    loss = loss + l

                loss.backward()
                return loss

            optimizer.step(closure)
            
        if batch_idx % args.log_interval == 0:

            with torch.no_grad():
                output = model(data)
                if args.loss == "cross_entropy":
                    losses = [F.cross_entropy(output, target)]
                elif args.loss == "mse":
                    losses = [F.mse_loss(output, target)]
                else:
                    raise NotImplementedError

                logger_info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: '.format(
                    epoch, (batch_idx - start_index) * len(data), len(train_loader.dataset),
                    100. * (batch_idx - start_index) / len(train_loader)) + ",".join([str(l.item()) for l in losses])
                logger.info(logger_info)

        if args.save_model_along and (batch_idx + 1) % args.save_model_interval == 0:
            torch.save(model.state_dict(), f"{args.exp_id}/{args.operation}_{batch_idx + 1}.pt")
            logger.info(f"model was saved to {args.exp_id}/{args.operation}_{batch_idx + 1}.pt")

        if args.dry_run:
            break

    return model

def test(args, model, device, test_loader, logger, name):
    model.eval()

    if args.loss == "cross_entropy":
        
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = todevice(data, device), todevice(target, device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        logger.info("\t"+name+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return 100. * correct / len(test_loader.dataset)
    
    elif args.loss == "mse":
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = todevice(data, device), todevice(target, device)
                output = model(data)
                per_sample_loss = F.mse_loss(output, target, reduction='none')
                per_sample_rmse = torch.sqrt(per_sample_loss)
                test_loss += per_sample_rmse.sum().item()  # sum up batch loss

        test_loss /= len(test_loader.dataset)

        logger.info("\t"+name+' set: Average loss: {:.6f}'.format(test_loss))

        return test_loss
    
    else:
        raise NotImplementedError

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--model', type=str, default="KAN", #required=True,
                        help='network structure')
    parser.add_argument('--layers_width', type=int, default=[5], nargs='+', #required=True,
                        help='the width of each hidden layer')
    parser.add_argument('--batch_norm', action='store_true', default=False,
                        help='whether use batch normalization')
    parser.add_argument('--activation_name', type=str, default="gelu", 
                        help='activation function')
    parser.add_argument('--pre_train_ckpt', type=str, default="", 
                        help='path of the pretrained model')

    parser.add_argument('--dataset', type=str, default="FMNIST", #required=True,
                        help='dataset')

    parser.add_argument('--batch-size', type=int, default=1024,
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, # 100 MNIST pretrain, 5 Finetune
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='supported optimizer: adam, lbfgs')
    # parser.add_argument('--gamma', type=float, default=0.7,
    #                     help='Learning rate step gamma (default: 0.7, 1.0 for fewshot)')
    parser.add_argument('--loss', type=str, default="cross_entropy",
                        help='loss function')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1314,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model-interval', type = int, default=-1, 
                        help='whether save model along training')
    ################# Parameters for KAN #################
    parser.add_argument('--kan_bspline_grid', type=int, default=5, 
                        help='the grid size of the bspline in the KAN layer')
    parser.add_argument('--kan_bspline_order', type=int, default=3, 
                        help='the order of the bspline in the KAN layer')
    parser.add_argument('--kan_shortcut_name', type=str, default="silu", 
                        help='the shortcut(base) function in the KAN layer: zero, identity, silu')
    parser.add_argument('--kan_grid_range', type=float, default=[-1, 1], nargs=2,
                        help='the range of the grid in the KAN layer. default is [-1, 1]. but for general normalized data, it can be larger.')
    # parser.add_argument('--num_KAFs', type=int, default=3,
    #                     help='the number of KAFs in the KANMoE layer')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='the number of classes in the KAF layer of output layer')
    parser.add_argument('--input_size', type=int, default=28*28,
                        help='the input size of the KAF layer of input layer')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='the dropout probability in the KAF layer')
    # parser.add_argument('--topk', type=int, default=2, 
    #                     help='the topk of the KANMoE layer')
    parser.add_argument('--num_grids', type=int, default=8, 
                        help='num of BSF fuction')
    parser.add_argument('--no_use_SVD', action='store_true', default=False,
                        help='whether use SVD in the KANMoE layer')
    parser.add_argument('--use_grouped_params', action='store_true', default=True,
                        help='whether to use grouped parameter sharing')
    parser.add_argument('--group_linear_rank', type=int, default=None,
                        help='rank of the linear layer in the group')
    parser.add_argument('--spline_dropout', type=float, default=0.1,
                        help='dropout probability for the spline layer')
    parser.add_argument('--act_exp', type=float, default=1.64,
                        help='activation_expectation of different base_activation')
    ################# Parameters for KAN #################
    ################# Parameters for MLP #################
    ## pass ##
    ################# Parameters for MLP #################
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    randomness_control(args.seed)

    args.save_model_along = args.save_model_interval > 0

    args.exp_id = f"../logs/{args.dataset}/{args.model}/"
    args.exp_id = args.exp_id + f"{'_'.join([str(w) for w in args.layers_width])}__{args.batch_norm}__{args.activation_name}"
    args.exp_id = args.exp_id + f"__{args.batch_size}__{args.epochs}__{args.lr}__{args.seed}__{args.optimizer}__{args.group_linear_rank}"
    os.makedirs(args.exp_id, exist_ok = True)
    ################# id for KAN #################
    if args.model in ["KAN", "KAN_Text", "KAN_MOE_SHARE"]:
        args.exp_id = args.exp_id + f"/{args.kan_bspline_grid}__{args.kan_bspline_order}__{args.kan_shortcut_name}"+ f"__{'_'.join([str(w) for w in args.kan_grid_range])}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for KAN #################
    ################# id for KANMoE #################
    elif args.model in ["KAF", "KAF_Text"]:
        args.exp_id = args.exp_id + f"/{args.num_KAFs}__{args.dropout_prob}__{args.num_classes}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for KANMoE #################
    ################# id for GPKAN #################
    elif args.model in ["GPKAN", "GPKAN_Text"]:
        args.exp_id = args.exp_id + f"/{args.num_KAFs}__{args.dropout_prob}__{args.num_classes}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for GPKAN #################
    ################# id for FAN #################
    elif args.model in ["FAN", "FAN_Text"]:
        args.exp_id = args.exp_id + f"/{args.num_KAFs}__{args.dropout_prob}__{args.num_classes}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for FAN #################  
    ################# id for BSpline MLP #################
    elif args.model == "BSpline_MLP":
        args.exp_id = args.exp_id + f"/{args.kan_bspline_grid}__{args.kan_bspline_order}"+ f"__{'_'.join([str(w) for w in args.kan_grid_range])}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for BSpline MLP #################
    ################# id for BSpline First MLP #################
    elif args.model == "BSpline_First_MLP":
        args.exp_id = args.exp_id + f"/{args.kan_bspline_grid}__{args.kan_bspline_order}"+ f"__{'_'.join([str(w) for w in args.kan_grid_range])}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for BSpline First MLP #################
    ################# id for MLP #################
    elif args.model in ["MLP", "MLP_Text"]:
        args.exp_id = args.exp_id + f"/default"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for MLP #################
    else:
        raise NotImplementedError
    
    if os.path.exists(os.path.join(args.exp_id, "log")):
        with open(os.path.join(args.exp_id, "log"), "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                if "training process was finished" in lines[-1]:
                    raise ValueError("training process was finished")

    logger, formatter = get_logger(args.exp_id, None, "log", level=logging.INFO)

    train_loader, test_loader, num_classes, input_size = get_loader(args, use_cuda = use_cuda)

    args.output_size = num_classes
    args.input_size = input_size

    args.activation = get_activation(args)
    args.kan_shortcut_function = get_shortcut_function(args)

    model = get_model(args)
    logger.info(model)
    num_parameters, flops = get_model_complexity(model, logger, args)
    model = model.to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-3)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == "lbfgs":
        optimizer = LBFGS(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            history_size=10, 
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-32, 
            tolerance_change=1e-32, 
            tolerance_ys=1e-32)
    else:
        raise NotImplementedError

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.loss == "cross_entropy":
        best_test_metric = 0 
    elif args.loss == "mse":
        best_test_metric = 1e10 
    else:
        raise NotImplementedError
    corresponding_train_metric = 0

    fvctimer = Timer()
    for epoch in range(1, args.epochs + 1):
        fvctimer.reset()  
        train(args, model, device, train_loader, optimizer, epoch, logger = logger, start_index = (epoch - 1) *len(train_loader))
        epoch_time = fvctimer.seconds()  
        logger.info(f"Epoch {epoch} training time: {epoch_time:.2f} seconds") 
        
        train_metric = test(args, model, device, train_loader, logger = logger, name = "train")
        test_metric = test(args, model, device, test_loader, logger = logger, name = "test")
        
        if args.loss == "cross_entropy":
            if test_metric > best_test_metric:
                best_test_metric = test_metric
                corresponding_train_metric = train_metric
        elif args.loss == "mse":
            if test_metric < best_test_metric:
                best_test_metric = test_metric
                corresponding_train_metric = train_metric
        else:
            raise NotImplementedError


        # scheduler.step()

    total_training_time = fvctimer.seconds()
    average_training_time_per_epoch = fvctimer.avg_seconds()
    logger.info(f"total training time: {total_training_time:,} seconds; average training time per epoch: {average_training_time_per_epoch:,} seconds")

    write_results(
        args,
        train_metric = corresponding_train_metric,
        test_metric = best_test_metric,
        num_parameters = num_parameters,
        flops = flops,
        total_training_time = total_training_time,
        average_training_time_per_epoch = average_training_time_per_epoch,
        group_linear_rank = args.group_linear_rank
    )

    if args.save_model:
        torch.save(
            {   
                "args" : args,
                "state_dict" : model.state_dict(),
                "metrics" : {
                    "train_metric" : corresponding_train_metric,
                    "test_metric" : best_test_metric,
                    "num_parameters" : num_parameters,
                    "flops" : flops,
                    "total_training_time" : total_training_time,
                    "average_training_time_per_epoch" : average_training_time_per_epoch
                }
            }, f"{args.exp_id}/ckpt.pt")
        logger.info(f"model was saved to {args.exp_id}/ckpt.pt")

    logger.info(f"training process was finished")

if __name__ == '__main__':
    main()

import torch
from model import ResNet18, VGG11Model
import torch.optim as optim 
import copy
import random 
import numpy as np
import time 
import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def reset_model_to_zero(model):
    for param in model.parameters():
        param.data.fill_(0.0)

def federated_train(trainloaders, valloaders, testloader, config):
    model = ResNet18(num_classes=2)
    # model = VGG11Model(num_classes=2)
    nets = {net_i: copy.deepcopy(model) for net_i in range(len(trainloaders))}
    global_model = copy.deepcopy(model)  # Bản sao mô hình toàn cục
    c_nets = {net_i: copy.deepcopy(model) for net_i in range(len(trainloaders))}  # Bản sao mô hình trên từng client
    c_global = copy.deepcopy(model)

    # c_global_para = c_global.state_dict()
    # for net_id, net in c_nets.items():
    #     net.load_state_dict(c_global_para)
    for c_net in c_nets.values():
        reset_model_to_zero(c_net)
    reset_model_to_zero(c_global)

    num_rounds = config.num_rounds  # Số vòng huấn luyện
    accs = []
    accs.append(evaluate(global_model, testloader))
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        start = time.time()
        global_para = global_model.state_dict()

        # Chọn các client tham gia vào mỗi round
        selected_clients = select_clients(trainloaders, config.clients_per_round)
        
        # Huấn luyện trên các client đã chọn
        for client in selected_clients:
            nets[client].load_state_dict(global_para)
        local_train_net_scaffold(nets, selected_clients, global_model, c_nets, c_global, config, trainloaders, device=DEVICE)

        total_data_points = sum([len(trainloaders[client].dataset) for client in selected_clients])
        freqs = [len(trainloaders[client].dataset) / total_data_points for client in selected_clients]

        for idx in range(len(selected_clients)):
            net_para = nets[selected_clients[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * freqs[idx]
        global_model.load_state_dict(global_para)
        global_model.to('cpu')
        acc = evaluate(global_model, testloader)        
        accs.append(acc)
        end = time.time()
        print(f'Time for round {round_num + 1}: ', end-start)
    plot_accuracy(accs)



def local_train_net_scaffold(nets, selected_clients, global_model, c_nets, c_global, config, trainloaders, device='cpu'):
    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id in selected_clients:
        net = nets[net_id]
        net.to(device)

        c_nets[net_id].to(device)

        c_delta_para = train_net_scaffold(net, global_model, c_nets[net_id], c_global, trainloaders[net_id], config, device=device)
        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]
        
    for key in total_delta:
        # total_delta[key] /= len(selected_clients) ### ???
        total_delta[key] /= config.num_clients
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    # nets_list = list(nets.values())
    # return nets_list
def train_net_scaffold(net, global_model, c_local, c_global, trainloader, config, device):
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=config.learning_rate,
        momentum=config.momentum
    )
    critierion = torch.nn.CrossEntropyLoss().to(device)
    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()
    cnt = 0
    for _ in range(config.num_epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = critierion(output, target)
            loss.backward()
            optimizer.step()

            net_para = net.state_dict()
            for key in net_para:
                net_para[key] = net_para[key] - config.learning_rate * (c_global_para[key] - c_local_para[key])
            net.load_state_dict(net_para)
            cnt += 1
    
    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * config.learning_rate)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    net.to('cpu')
    return c_delta_para

            
        
def plot_accuracy(accs):
    print('accuracies: ', accs)
    num_rounds = len(accs)-1
    plt.plot(range(0, num_rounds + 1), accs, marker='o', label='Accuracy')
    plt.xlabel('Round')
    plt.xticks(range(0, num_rounds + 1))
    plt.ylabel('Accuracy')
    plt.title('Scaffold on ResNet18 over Rounds')
    plt.grid(True)
    plt.legend()
    plt.savefig('running_outputs/accuracy_summary.png')
    plt.close()



def select_clients(trainloaders, clients_per_round):
    """Chọn ngẫu nhiên một số client tham gia huấn luyện trong mỗi round."""
    # Số lượng client có sẵn
    total_clients = len(trainloaders)
    # Chọn ngẫu nhiên một số client
    selected_clients = random.sample(range(total_clients), clients_per_round)
    return selected_clients


def evaluate(model, testloader):
    """Đánh giá mô hình trên tập kiểm tra."""
    model.eval()  # Chuyển sang chế độ đánh giá
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

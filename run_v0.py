import torch
from torch.utils.data import DataLoader
from model.HierRec import HierRec
from utils.data_process import ReadDataset
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
import time



def add_list(a,b):
    if b.dim()!=0:
        a.extend(b.tolist())
    else:
        a.append(b.tolist())
    return a

class EarlyStopper(object):

    def __init__(self, num_trials, save_path, is_max=True):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.is_max = is_max
        if is_max:
            self.best_accuracy = 0.0
        else:
            self.best_accuracy = 100000000000.0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if self.is_max:
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.trial_counter = 0
                torch.save(model.state_dict(), self.save_path)
                return True
            elif self.trial_counter + 1 < self.num_trials:
                self.trial_counter += 1
                return True
            else:
                return False
        else:
            if accuracy < self.best_accuracy:
                self.best_accuracy = accuracy
                self.trial_counter = 0
                torch.save(model.state_dict(), self.save_path)
                return True
            elif self.trial_counter + 1 < self.num_trials:
                self.trial_counter += 1
                return True
            else:
                return False

def model_test(model, loader, domain_num, device, each_domain=False):
    model.eval()
    predictions = []
    labels = []
    if each_domain:
        predictions_domain = [[] for i in range(domain_num)]
        labels_domain = [[] for i in range(domain_num)]
    for field, label in loader:
        field, label = field.to(device), label.float().to(device)
        pred = model(field)
        predictions = add_list(predictions, pred)
        labels = add_list(labels, label)
        if each_domain:
            for domain_idx in range(domain_num):
                predictions_domain[domain_idx] = add_list(predictions_domain[domain_idx], pred[torch.argwhere(field[:,0]==domain_idx).squeeze()])
                labels_domain[domain_idx] = add_list(labels_domain[domain_idx], label[torch.argwhere(field[:,0]==domain_idx).squeeze()])
    auc = roc_auc_score(labels, predictions)
    logloss = log_loss(labels, predictions)
    if each_domain:
        auc_domain = []
        logloss_domain = []

        for i in range(domain_num):
            # print("i: {}, pre: {}, label: {}".format(i, len(predictions_domain[i]), len(labels_domain[i])))
            auc_domain.append(roc_auc_score(labels_domain[i], predictions_domain[i]))
            logloss_domain.append(log_loss(labels_domain[i], predictions_domain[i]))
        # auc_domain = [roc_auc_score(labels_domain[i], predictions_domain[i]) for i in range(domain_num)]
        # logloss_domain = [log_loss(labels_domain[i], predictions_domain[i]) for i in range(domain_num)]
        return auc, auc_domain, logloss, logloss_domain
    return auc, logloss


def model_train(model, train_loader, val_loader, optimizer, criterion,
                epoch, save_dir, domain_num, device, test_loader=None):
    early_stopper = EarlyStopper(
        num_trials=2, save_path=save_dir, is_max=False)
    for ep in range(epoch):
        # train
        model.train()
        train_loader = tqdm(train_loader, mininterval=1)
        train_loader.set_description('Epoch %i' % ep)
        for field, label in train_loader:
            field, label = field.to(device), label.float().to(device)
            pred = model(field)
            loss = criterion(pred, label).mean()
            model.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            train_loader.set_postfix(loss=loss.detach().item())
        # if ind % 500 == 0:
        #     print("Loss: ", loss.item())

        auc, logloss = model_test(model, val_loader, domain_num, device)
        print("Epoch {}, AUC: {}, Logloss: {}".format(ep, auc, logloss))
        time.sleep(0.001)
        if not early_stopper.is_continuable(model, logloss):
            print('Validation: best logloss: {}'.format(early_stopper.best_accuracy))
            break

    model.load_state_dict(torch.load(save_dir))
    if test_loader is not None:
        print("----------Start testing...----------")
        auc, auc_domain, logloss, logloss_domain = model_test(model, test_loader, domain_num, device, each_domain=True)
        print('Test auc: {}, logloss: {}'.format(auc, logloss))
        print('Test auc/domain: {}'.format(auc_domain))
        print('Test logloss/domain: {}'.format(logloss_domain))
        return auc, auc_domain, logloss, logloss_domain, model
    return early_stopper.best_auc, model


def main(model_name,
         dataset_name,
         data_dir,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         dropout,
         device,
         save_dir,
         mlp_dims,
         embed_dims,
         im_num,):
    print("Model: ", model_name)
    print("Dataset: ", dataset_name)
    print("Data Dir: ", data_dir)
    print("Epoch: ", epoch)
    print("Learning Rate: ", learning_rate)
    print("Batch Size: ", batch_size)
    print("Weight_Decay: ", weight_decay)
    print("Embed_dims: ", embed_dims)
    print("Mlp_dims: ", mlp_dims)
    print("Head Num: ", im_num)
    print("Dropout: ", dropout)
    print("Device: ", device)
    print("Save Dir: ", save_dir)


    print("----------Preparing dataset and model...----------")
    device = torch.device(device)
    if dataset_name == 'aliccp':
        train_set = ReadDataset(data_dir=data_dir, name=dataset_name, train='train')
        train_set, val_set = train_set.split([0.9])
        test_set = ReadDataset(data_dir=data_dir, name=dataset_name, train='test')
    elif dataset_name == 'douban':
        train_set = ReadDataset(data_dir=data_dir, name=dataset_name, train='train')
        val_set = ReadDataset(data_dir=data_dir, name=dataset_name, train='val')
        test_set = ReadDataset(data_dir=data_dir, name=dataset_name, train='test')
    elif dataset_name == 'kuairand':
        train_set = ReadDataset(data_dir=data_dir, name=dataset_name)
        train_set, val_set, test_set = train_set.split([0.8, 0.1])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    print("Num of samples: Train: {}, Val: {}, Test: {}"
          .format(len(train_set), len(val_set), len(test_set)))
    train_dims = train_set.field_dims()
    val_dims = val_set.field_dims()
    test_dims = test_set.field_dims()
    field_dims = [max(train_dims[i], val_dims[i], test_dims[i]) for i in range(len(train_dims))]
    domain_num = field_dims[0]

    if model_name=="hierrec":
        model = HierRec(field_dims, embed_dim=embed_dims, mlp_dims=mlp_dims,
                     dropout=dropout, im_num=im_num, device=device).to(device)


    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCELoss()

    print("----------Start training&testing...----------")
    time.sleep(0.1)
    start_time = time.time()
    auc,_,logloss,_,_ = model_train(model, train_loader, val_loader, optimizer, criterion,
                epoch, save_dir, domain_num, device, test_loader=test_loader)
    end_time = time.time()
    print("Time Spent: {} second".format(end_time-start_time))
    return auc, logloss




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="hierrec")
    parser.add_argument('--dataset_name', default="kuairand")
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--embed_dims', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--im_num', type=int, default=4)
    parser.add_argument('--mlp_dims', type=list, default=[[64], [16, 64], [16, 64], [64]])
    parser.add_argument(
        '--device', default='cuda:0')
    parser.add_argument('--save_dir', default='log/model.pt')
    parser.add_argument('--repeat_experiments', type=int, default=1)
    args = parser.parse_args()


    auc_all = 0.0
    logloss_all = 0.0
    for i in range(args.repeat_experiments):
        print("--------------------Experiment: {}--------------------".format(i))
        auc_tmp, logloss_tmp = main(args.model_name,
             args.dataset_name,
             args.data_dir,
             args.epoch,
             args.learning_rate,
             args.batch_size,
             args.weight_decay,
             args.dropout,
             args.device,
             args.save_dir,
             args.mlp_dims,
             args.embed_dims,
             args.im_num,)
        auc_all+=auc_tmp
        logloss_all+=logloss_tmp
        print("Average AUC: {}, Logloss: {}".format(auc_all/(i+1), logloss_all/(i+1)))

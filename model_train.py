
from utils import  timer_simple, get_snp_emb_table, sparse_regular
import logging
import torch
from torch import nn 
from torch.utils.data import DataLoader
from collections import Counter

    


# @timer_simple
def do_train(data_loader, model, loss_fn, optimizer, global_step, print_batches = 30, regular_weight=1e-5
             , device=None, test_flag=False):
    logging.debug('dataset_batches->{}'.format(len(data_loader)))

    def train_step(x, y):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred = torch.squeeze(pred)
        y = torch.squeeze(y)
        loss = loss_fn(pred, y) 
        regular_loss = 0
        if regular_weight > 0:
            emb_param = get_snp_emb_table(model)
            if emb_param is not None:
                regular_loss = sparse_regular(emb_param, weight=1)
        total_loss = loss + regular_weight*regular_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss, regular_loss

    for num_batches, (x, y) in enumerate(data_loader):
        global_step += 1
        loss, regular_loss = train_step(x, y)
        if global_step%print_batches == 1:
            y_dist = y.detach().cpu().numpy().flatten()
            y_dist = Counter(y_dist)
            logging.info('after global_step->{}, regular_loss->{:4f} loss -> {:4f}, label_dist -> {}'.format(
                global_step, regular_loss, loss, y_dist))

        if test_flag and num_batches > 3:
            break

    return global_step


def train_a_model(device, model_class, dataset, train_conf):
    model = model_class()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_conf['lr'])
    train_loader = DataLoader( dataset=dataset, batch_size=train_conf['batch_size'], drop_last=True, shuffle=True)
    model.train()
    global_step = 0
    logging.info('begin train model_class {}, ds {}'.format(model_class, len(dataset)))
    for n_epoch in range(train_conf['epoch']):
        global_step = do_train(train_loader, model, loss_fn, optimizer, global_step
                    , regular_weight=0.0
                    , device=device
                    , test_flag=train_conf.get('test_flag', False)
                    , print_batches=train_conf.get('print_global_step', 60) 
                    ) 
    return model

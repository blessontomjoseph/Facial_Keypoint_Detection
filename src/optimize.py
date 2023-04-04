import torch
import wandb
import dataloader
import model
from tqdm.autonotebook import tqdm,trange

sweep_config={'method':'bayes'}
metric={'name':'loss',
        'goal':'minimize'
       }
sweep_config['metric']=metric
parameters_dict={'learning_rate':{'values':[10**(-i) for i in range(1,11)]},
                 
                 'train_batch_size':{'values':[16]},
                  
                 'val_batch_size':{'values':[8]},
                 
                 'epochs':{'values':[3]},
                 
                 'optim':{'values':['sgd','adam','adamw']},
                 
                 'criterion':{'values':['l2']},
                 
                 'scheduler':{'values':['cyclic','one_cycle','sgd_wr']},
                 
                 'momentum':{'distribution':'uniform',
                            'min':0.1,
                            'max':0.9}    
                 
                }


sweep_config['parameters']=parameters_dict
sweep_id=wandb.sweep(sweep_config,project='facial-key')

def train(config=None):
    with wandb.init(config=config):
        config=wandb.config
        facial_key_model=model.model_init()
        train_loader,val_loader=dataloader.data_init(config.train_batch_size,config.val_batch_size)
        criterion=model.criterion_init(config.criterion)
        optimizer=model.optimizer_init(model,config.optim,config.learning_rate,momentum=config.momentum)
        scheduler=model.scheduler_init(config.scheduler,optimizer,train_loader,config.epochs)
        
        epochs=config.epochs
        
        t_l,v_l=[],[]
        for epoch in trange(epochs):
            losses=0
            for batch in train_loader:
                x=batch['x'].to('cuda')
                y=batch['y'].to('cuda')
                out=model(x)
                loss=criterion(out,y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses+=loss.item()
            t_l.append(losses/len(train_loader))
            wandb.log({'train_loss':t_l[-1]})

            with torch.no_grad():
                losses=0
                for batch in val_loader:
                    x=batch['x'].to('cuda')
                    y=batch['y'].to('cuda')
                    out=model(x)
                    loss=criterion(out,y)
                    losses+=loss.item()
                v_l.append(losses/len(val_loader))
                wandb.log({'loss':v_l[-1]})                        

    return t_l,v_l

wandb.agent(sweep_id,train,count=10)
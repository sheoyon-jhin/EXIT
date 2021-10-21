import copy
import json
import math
import numpy as np
import os
import pathlib
import sklearn.metrics
import torch
import tqdm

import models

here = pathlib.Path(__file__).resolve().parent
BASELINE_MODELS =['ncde_forecasting']

def _add_weight_regularisation(loss_fn, regularise_parameters,writer,scaling=0.03):
    def new_loss_fn(pred_y, true_y,epoch = None,mode=None):
        task_loss = loss_fn(pred_y, true_y)
        reg_loss = 0
        for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                reg_loss += scaling * parameter.norm()
        writer.add_scalar(f'{mode}/Before_Regularise_loss',task_loss, epoch)
        writer.add_scalar(f'{mode}/l2_Regularise_loss',reg_loss, epoch)
        return reg_loss + task_loss
    return new_loss_fn


class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        out, reg = self.model(*args, **kwargs)
        return out.squeeze(-1), reg


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class _AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]


def _evaluate_metrics(name,dataloader, model, times,terminal_time, loss_fn,mode,epoch, num_classes, device, kwargs):
    with torch.no_grad():
        total_accuracy = 0
        total_confusion = torch.zeros(num_classes, num_classes).numpy()  # occurs all too often
        total_dataset_size = 0
        total_loss = 0
        true_y_cpus = []
        pred_y_cpus = []
        thresholded_y_cpus=[]

        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)
            # term_time = torch.arange(0,terminal_time)
            # import pdb ;pdb.set_trace()
            
            pred_y, _ = model(times,terminal_time, coeffs, lengths, **kwargs)

            if num_classes == 2:
                thresholded_y = (pred_y > 0).to(true_y.dtype)
            else:
                thresholded_y = torch.argmax(pred_y, dim=1)
            true_y_cpu = true_y.detach().cpu()
            pred_y_cpu = pred_y.detach().cpu()
            thresholded_y_cpu = thresholded_y.detach().cpu()
            if num_classes == 2:
                # Assume that our datasets aren't so large that this breaks
                true_y_cpus.append(true_y_cpu)
                pred_y_cpus.append(pred_y_cpu)
                thresholded_y_cpus.append(thresholded_y_cpu)
                
            

            total_accuracy += (thresholded_y == true_y).sum().to(pred_y.dtype)
            total_confusion += sklearn.metrics.confusion_matrix(true_y_cpu, thresholded_y_cpu,
                                                                labels=range(num_classes))
            total_dataset_size += batch_size
            total_loss += loss_fn(pred_y, true_y, epoch, mode) * batch_size

        total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
        total_accuracy /= total_dataset_size
        metrics = _AttrDict(accuracy=total_accuracy.item(), confusion=total_confusion, dataset_size=total_dataset_size,
                            loss=total_loss.item())

        if num_classes == 2:
            true_y_cpus = torch.cat(true_y_cpus, dim=0)
            pred_y_cpus = torch.cat(pred_y_cpus, dim=0)
            thresholded_y_cpus = torch.cat(thresholded_y_cpus,dim=0)
            metrics.auroc = sklearn.metrics.roc_auc_score(true_y_cpus, pred_y_cpus)
            metrics.average_precision = sklearn.metrics.average_precision_score(true_y_cpus, pred_y_cpus)
            # import pdb ; pdb.set_trace()
            metrics.f1score_macro = sklearn.metrics.f1_score(true_y_cpus,thresholded_y_cpus,average='macro')
            metrics.f1score_micro = sklearn.metrics.f1_score(true_y_cpus,thresholded_y_cpus,average='micro')
            metrics.f1score_weighted = sklearn.metrics.f1_score(true_y_cpus,thresholded_y_cpus,average='weighted')
            
            
        return metrics


def _evaluate_metrics_forecasting(name, dataloader, model, times, loss_fn, mode,epoch,num_classes, device, kwargs):
    with torch.no_grad():
        total_accuracy = 0
        # total_confusion = torch.zeros(num_classes, num_classes).numpy()  # occurs all too often
        total_dataset_size = 0
        total_loss = 0
        mse_loss = 0
        logpz_loss = 0
        true_y_cpus = []
        pred_y_cpus = []

        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)

            
            pred_y = model(times, coeffs, lengths, **kwargs)
            if 'google' in name:
                pred_y = pred_y[:,:,:-1]
                true_y = true_y[:,:,:-1]

            
            total_dataset_size += batch_size
            
            total_loss += loss_fn(pred_y, true_y, epoch, mode) * batch_size
            
            

        total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
        mse_loss /= total_dataset_size
        logpz_loss /= total_dataset_size
        
        
        metrics = _AttrDict( dataset_size=total_dataset_size,loss=total_loss.item())
    
        return metrics

class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        
        if exc_type is AssertionError:
            # import pdb ; pdb.set_trace()
            self.tqdm_range.write('Caught AssertionError: ' + str(exc_val))
            return True


def _train_loop(name,train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, max_epochs, num_classes,writer, device,
                kwargs, step_mode):
    
    model.train()
    best_model = model
    best_train_loss = math.inf
    best_train_accuracy = 0
    best_val_accuracy = 0
    best_val_aucroc = 0
    best_val_loss = math.inf
    best_train_accuracy_epoch = 0
    best_train_aucroc_epoch = 0
    
    best_train_loss_epoch = 0
    history = []
    breaking = False
    # import pdb ; pdb.set_trace()
    # if time_l =='front':
    #   fixed_start_time = times[-1]
    # elif time_l == 'end':
    #   fixed_start_time = times[0]
    
    
    
    if step_mode == 'trainloss':
        print("trainloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        print("patience:5")

    elif step_mode=='valloss':
        print("valloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        print("patience:5")

    elif step_mode == 'valaccuracy':
        print("valaccuracy")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')
        print("patience:5")

    elif step_mode=='valauc':
        print("valauc")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')
        print("patience:5")
        
    elif step_mode=='none':
        epoch_per_metric=1 
        plateau_terminate=50
        print("none")

    

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    iterations = 0
    for epoch in tqdm_range:
        if breaking:
            break
        for batch in train_dataloader:
            iterations += 1
            batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):
                
                *train_coeffs, train_y, lengths = batch
                
                if 'forecasting' in name:
                    
                    # import pdb ; pdb.set_trace()
                    pred_y= model(times, train_coeffs, lengths, **kwargs)
                    # print(pred_y.shape)
                    if 'google' in name:
                        pred_y = pred_y[:,:,:-1]
                        train_y = train_y[:,:,:-1]
                
                else:
                    # import pdb ; pdb.set_trace()
                    pred_y = model(times, train_coeffs, lengths, **kwargs)
                
                loss = loss_fn(pred_y, train_y,iterations,"trains")
                
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

                
                
        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            if 'forecasting' in name:
                train_metrics = _evaluate_metrics_forecasting(name,train_dataloader, model, times, loss_fn,'train',epoch, num_classes, device, kwargs)
                val_metrics = _evaluate_metrics_forecasting(name,val_dataloader, model, times, loss_fn,'val',epoch, num_classes, device, kwargs)
                test_metrics = _evaluate_metrics_forecasting(name,test_dataloader, model, times, loss_fn,'test',epoch, num_classes, device, kwargs)
                
                
                writer.add_scalar('validation/loss',val_metrics.loss,epoch)
                writer.add_scalar('test/loss',test_metrics.loss,epoch)
                
                
            else:
                train_metrics = _evaluate_metrics(name,train_dataloader, model, times, loss_fn,'train',epoch, num_classes, device, kwargs)
                val_metrics = _evaluate_metrics(name,val_dataloader, model, times, loss_fn,'val',epoch, num_classes, device, kwargs)
                test_metrics = _evaluate_metrics(name,test_dataloader, model, times, loss_fn,'test',epoch, num_classes, device, kwargs)

            # train_metrics = _evaluate_metrics(train_dataloader, model, times, loss_fn, num_classes, device, kwargs)
            # val_metrics = _evaluate_metrics(val_dataloader, model, times, loss_fn, num_classes, device, kwargs)
                writer.add_scalar('train/accuracy',train_metrics.accuracy,epoch)
                
                writer.add_scalar('validation/loss',val_metrics.loss,epoch)
                writer.add_scalar('validation/accuracy',val_metrics.accuracy,epoch)
                
                
                writer.add_scalar('test/loss',test_metrics.loss,epoch)
                writer.add_scalar('test/accuracy',test_metrics.accuracy,epoch)
                

                if num_classes == 2:
                    writer.add_scalar('train/AUROC',
                                    train_metrics.auroc, epoch)
                    writer.add_scalar('validation/AUROC',
                                    val_metrics.auroc, epoch)
                    writer.add_scalar(
                        'test/AUROC', test_metrics.auroc, epoch)
            
            
            model.train()
            if 'forecasting' in name : 
                if train_metrics.loss * 1.0001 < best_train_loss:
                    best_train_loss = train_metrics.loss
                    best_train_loss_epoch = epoch
                
                base_base_loc = here/'saved_models'
                loc = base_base_loc / name
                if not os.path.exists(base_base_loc):
                    os.mkdir(base_base_loc)
                if not os.path.exists(loc):
                    os.mkdir(loc)
                    
                if val_metrics.loss < best_val_loss:
                    print("Best Epoch: {} Val loss : {:.3} , Test loss {:.3} ".format(epoch,val_metrics.loss,test_metrics.loss))
                
                    best_val_loss = val_metrics.loss
                    del best_model  # so that we don't have three copies of a model simultaneously
                    best_model = copy.deepcopy(model)  
                    if val_metrics.loss <= 0.03:
                        print("Saving Model")
                        print(f"[Saved EPOCH {epoch}] Validation MSE loss : {val_metrics.loss} Test MSE loss : {test_metrics.loss}")
                        
                        PATH = os.path.dirname(os.path.abspath(__file__))

                        torch.save(best_model.state_dict(),PATH+"/saved_models/"+name+"/Epoch_"+str(epoch)+"_TestMSEloss_"+str(test_metrics.loss)+".pt")
                    
                
                tqdm_range.write('Epoch: {}  Train loss: {:.3}  Val loss: {:.3}  '
                                'Test loss : {:.3} '
                                ''.format(epoch, train_metrics.loss,  val_metrics.loss,
                                        test_metrics.loss))
            

                
            else:
                if train_metrics.loss * 1.0001 < best_train_loss:
                    best_train_loss = train_metrics.loss
                    best_train_loss_epoch = epoch

                if train_metrics.accuracy > best_train_accuracy * 1.001:
                    best_train_accuracy = train_metrics.accuracy
                    best_train_accuracy_epoch = epoch

            
                if num_classes ==2 :
                    
                    base_base_loc = here/'saved_models'
                    loc = base_base_loc / name

                    if not os.path.exists(base_base_loc):
                        os.mkdir(base_base_loc)
                    if not os.path.exists(loc):
                        os.mkdir(loc)
                    if val_metrics.auroc ==best_val_aucroc:
                        if test_metrics.auroc >= 0.82:
                            
                            print("Saving Model")
                            print(f"[EPOCH {epoch}] Validation AUROC : {val_metrics.auroc} Test AUROC : {test_metrics.auroc}")
                            PATH = os.path.dirname(os.path.abspath(__file__))

                            torch.save(best_model.state_dict(),PATH+"/saved_models/"+name+"/Epoch_"+str(epoch)+"_TestAUC_"+str(test_metrics.auroc)+".pt")
                        
                    if val_metrics.auroc > best_val_aucroc:
                        
                        best_val_aucroc =  val_metrics.auroc
                        del best_model  # so that we don't have three copies of a model simultaneously
                        best_model = copy.deepcopy(model)
                        print("BEST Epoch: {} Val loss : {:.3} Val AUC : {:.3} , Test loss : {:.3} Test AUC : {:.3}".format(epoch,val_metrics.loss,val_metrics.auroc,test_metrics.loss,test_metrics.auroc))
                    
                        if test_metrics.auroc >= 0.82:
                            
                            print("Saving Model")
                            print(f"[EPOCH {epoch}] Validation AUROC : {val_metrics.auroc} Test AUROC : {test_metrics.auroc}")
                            
                            PATH = os.path.dirname(os.path.abspath(__file__))

                            torch.save(best_model.state_dict(),PATH+"/saved_models/"+name+"/Epoch_"+str(epoch)+"_TestAUC_"+str(test_metrics.auroc)+".pt")
                            
                    print("Epoch: {} Val loss : {:.3} Val AUC : {:.3} , Test loss : {:.3} Test AUC : {:.3}".format(epoch,val_metrics.loss,val_metrics.auroc,test_metrics.loss,test_metrics.auroc))
                    tqdm_range.write('Epoch: {}  Train loss: {:.3}  Train AUC: {:.3} Train F1 score Macro {:.3} Train F1 score Micro {:.3} Train F1 score Weighted {:.3}  '
                                        'Val loss: {:.3} Val AUC: {:.3} Val F1 score Macro {:.3} Val F1 score Micro {:.3} Val F1 score Weighted {:.3} '
                                        'Test loss : {:.3} Test AUC : {:.3} Test F1 score Macro {:.3} Test F1 score Micro {:.3} Test F1 score Weighted {:.3}'
                                        ''.format(epoch, 
                                                train_metrics.loss, train_metrics.auroc,train_metrics.f1score_macro,train_metrics.f1score_micro,train_metrics.f1score_weighted, 
                                                val_metrics.loss, val_metrics.auroc,val_metrics.f1score_macro,val_metrics.f1score_micro,val_metrics.f1score_weighted,
                                                test_metrics.loss,test_metrics.auroc,test_metrics.f1score_macro,test_metrics.f1score_micro,test_metrics.f1score_weighted))


                else:
                    base_base_loc = here/'saved_models'
                    loc = base_base_loc / name
                    if not os.path.exists(base_base_loc):
                        os.mkdir(base_base_loc)
                    if not os.path.exists(loc):
                        os.mkdir(loc)
                    if val_metrics.accuracy ==best_val_accuracy:
                        print("BEST Epoch: {} Val loss : {:.3} Val Accuracy : {:.3} , Test loss {:.3} Test Accuracy : {:.3}".format(epoch,val_metrics.loss,val_metrics.accuracy,test_metrics.loss,test_metrics.accuracy))
                    
                        if test_metrics.accuracy >= 0.9:
                            
                            print("Saving Model")
                            print(f"[EPOCH {epoch}] Validation Accuracy : {val_metrics.accuracy} Test Accuracy : {test_metrics.accuracy}")
                            PATH = os.path.dirname(os.path.abspath(__file__))

                            torch.save(best_model.state_dict(),PATH+"/saved_models/"+name+"/Epoch_"+str(epoch)+"_TestAccuracy_"+str(test_metrics.accuracy)+".pt")
                        
                    
                    if val_metrics.accuracy > best_val_accuracy:
                        best_val_accuracy = val_metrics.accuracy
                        del best_model  # so that we don't have three copies of a model simultaneously
                        best_model = copy.deepcopy(model)
                        print("BEST Epoch: {} Val loss : {:.3} Val Accuracy : {:.3} , Test loss {:.3} Test Accuracy : {:.3}".format(epoch,val_metrics.loss,val_metrics.accuracy,test_metrics.loss,test_metrics.accuracy))
                    
                        if test_metrics.accuracy >= 0.9:
                            
                            print("Saving Model")
                            print(f"[EPOCH {epoch}] Validation Accuracy : {val_metrics.accuracy} Test Accuracy : {test_metrics.accuracy}")
                            
                            PATH = os.path.dirname(os.path.abspath(__file__))

                            torch.save(best_model.state_dict(),PATH+"/saved_models/"+name+"/Epoch_"+str(epoch)+"_TestAccuracy_"+str(test_metrics.accuracy)+".pt")
                        
                    tqdm_range.write('Epoch: {}  Train loss: {:.3}  Train accuracy: {:.3}  Val loss: {:.3}  '
                                    'Val accuracy: {:.3} Test loss : {:.3} Test accuracy {:.3} '
                                    ''.format(epoch, train_metrics.loss, train_metrics.accuracy, val_metrics.loss,
                                            val_metrics.accuracy,test_metrics.loss,test_metrics.accuracy))
                # if step_mode:
            #     scheduler.step(train_metrics.loss)
            # else:
            #     scheduler.step(val_metrics.accuracy)


            if step_mode == 'trainloss':
                scheduler.step(train_metrics.loss)
            elif step_mode=='valloss':
                scheduler.step(val_metrics.loss)
            elif step_mode == 'valaccuracy':
                scheduler.step(val_metrics.accuracy)
            elif step_mode=='valauc':
                scheduler.step(val_metrics.auroc)

            history.append(_AttrDict(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))

            if epoch > best_train_loss_epoch + plateau_terminate:
                tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
                                 ''.format(plateau_terminate))
                breaking = True
            # if epoch > best_train_accuracy_epoch + plateau_terminate:
            #     tqdm_range.write('Breaking because of no improvement in training accuracy for {} epochs.'
            #                      ''.format(plateau_terminate))
            #     breaking = True

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
    
    return history
    # return history


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def _save_results(name, result):
    loc = here / 'results' / name
    if not os.path.exists(loc):
        os.mkdir(loc)
    num = -1
    for filename in os.listdir(loc):
        try:
            num = max(num, int(filename))
        except ValueError:
            pass
    result_to_save = result.copy()
    del result_to_save['train_dataloader']
    del result_to_save['val_dataloader']
    del result_to_save['test_dataloader']
    result_to_save['model'] = str(result_to_save['model'])

    num += 1
    with open(loc / str(num), 'w') as f:
        json.dump(result_to_save, f, cls=_TensorEncoder)


def main(name, times,weight_decay, train_dataloader, val_dataloader, test_dataloader, device, make_model, num_classes, max_epochs,
         lr,time_lr, writer,kwargs, step_mode,learn_t,time_l, pos_weight=torch.tensor(1)):
    
    times = times.to(device)
    # import pdb;pdb.set_trace()
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None

    model, regularise_parameters = make_model()
    if 'forecasting' in name:
        # import pdb ; pdb.set_trace()
        loss_fn = torch.nn.functional.mse_loss
    else:
        if num_classes == 2:
            model = _SqueezeEnd(model)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        
        else:
            loss_fn = torch.nn.functional.cross_entropy
    
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters,writer)
    model.to(device)
    
    
    
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    
    
    history = _train_loop(name,train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, max_epochs,
                          num_classes,writer, device, kwargs, step_mode)
    
    # model.eval()
    
    # train_metrics = _evaluate_metrics(train_dataloader, model, times,terminal_time, loss_fn, num_classes, device, kwargs)
    # val_metrics = _evaluate_metrics(val_dataloader, model, times,terminal_time, loss_fn, num_classes, device, kwargs)
    # test_metrics = _evaluate_metrics(test_dataloader, model, times,terminal_time, loss_fn, num_classes, device, kwargs)

    if device != 'cpu':
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
    else:
        memory_usage = None
    print(f"memory_usage : {memory_usage}")
    result = _AttrDict(times=times,
                       memory_usage=memory_usage,
                       baseline_memory=baseline_memory,
                       num_classes=num_classes,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader,
                       model=model.to('cpu'),
                       parameters=_count_parameters(model),
                       history=history)
                    #    train_metrics=train_metrics,
                    #    val_metrics=val_metrics,
                    #    test_metrics=test_metrics)
    if name is not None:
        _save_results(name, result)
    return result


def make_model(name, input_channels, output_channels, hidden_channels,hi_hidden_channels, hidden_hidden_channels, num_hidden_layers,
               use_intensity,method,kinetic_energy_coef, jacobian_norm2_coef, div_samples, initial,output_time = 0):
    if name == 'ncde':
        def make_model():
            vector_field = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'idea4':
        def make_model():
            func_k = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_g = models.FinalTanh_g(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_f = models.ODEFunc_f(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_IDEA4(func=func_k,func_g = func_g,func_f = func_f, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels,method=method,
                                     kinetic_energy_coef=kinetic_energy_coef, jacobian_norm2_coef=jacobian_norm2_coef, 
                                     div_samples=div_samples, initial=initial)
                        
            return model, func_k
    elif name == 'idea4_2':
        def make_model():
            func_k = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_g = models.FinalTanh_g2(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_f = models.ODEFunc_f2(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_IDEA4(func=func_k,func_g = func_g,func_f = func_f, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels,method=method,
                                     kinetic_energy_coef=kinetic_energy_coef, jacobian_norm2_coef=jacobian_norm2_coef, 
                                     div_samples=div_samples, initial=initial)
            return model, func_k
    elif name == 'idea4_forecasting':
        def make_model():
            func_k = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_g = models.FinalTanh_g(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_f = models.ODEFunc_f(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_IDEA4_forecasting(func=func_k,func_g = func_g,func_f = func_f, input_channels=input_channels, hidden_channels=hidden_channels,
                                     hidden_hidden_channels=hidden_hidden_channels,output_channels=output_channels,method=method,
                                     kinetic_energy_coef=kinetic_energy_coef, jacobian_norm2_coef=jacobian_norm2_coef, 
                                     div_samples=div_samples, output_time = output_time,initial=initial)
            return model, func_k
    elif name == 'idea4_g_elu':
        def make_model():
            func_k = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_g = models.FinalTanh_g_elu(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_f = models.ODEFunc_f(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_IDEA4(func=func_k,func_g = func_g,func_f = func_f, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels,method=method,
                                     kinetic_energy_coef=kinetic_energy_coef, jacobian_norm2_coef=jacobian_norm2_coef, 
                                     div_samples=div_samples, initial=initial)
            return model, func_k
    elif name == 'idea4_g_elu_forecasting':
        def make_model():
            func_k = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_g = models.FinalTanh_g_elu(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_f = models.ODEFunc_f(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_IDEA4_forecasting(func=func_k,func_g = func_g,func_f = func_f, input_channels=input_channels, hidden_channels=hidden_channels,
                                     hidden_hidden_channels=hidden_hidden_channels,output_channels=output_channels,method=method,
                                     kinetic_energy_coef=kinetic_energy_coef, jacobian_norm2_coef=jacobian_norm2_coef, 
                                     div_samples=div_samples, output_time = output_time,initial=initial)
            return model, func_k
    elif name == 'idea4_forecasting2':
        def make_model():
            func_k = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_g = models.FinalTanh_g2(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_f = models.ODEFunc_f(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_IDEA4_forecasting(func=func_k,func_g = func_g,func_f = func_f, input_channels=input_channels, hidden_channels=hidden_channels,
                                     hidden_hidden_channels=hidden_hidden_channels,output_channels=output_channels,method=method,
                                     kinetic_energy_coef=kinetic_energy_coef, jacobian_norm2_coef=jacobian_norm2_coef, 
                                     div_samples=div_samples, output_time = output_time,initial=initial)
            return model, func_k
    elif name == 'idea4_forecasting3':
        def make_model():
            func_k = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_g = models.FinalTanh_g(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_f = models.ODEFunc_f2(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_IDEA4_forecasting(func=func_k,func_g = func_g,func_f = func_f, input_channels=input_channels, hidden_channels=hidden_channels,
                                     hidden_hidden_channels=hidden_hidden_channels,output_channels=output_channels,method=method,
                                     kinetic_energy_coef=kinetic_energy_coef, jacobian_norm2_coef=jacobian_norm2_coef, 
                                     div_samples=div_samples, output_time = output_time,initial=initial)
            return model, func_k
    elif name == 'gruode':
        def make_model():
            vector_field = models.GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels,
                                     hidden_channels=hidden_channels, output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'dt':
        def make_model():
            model = models.GRU_dt(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'decay':
        def make_model():
            model = models.GRU_D(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'odernn':
        def make_model():
            model = models.ODERNN(input_channels=input_channels, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'ncde_forecasting':
         def make_model():
            vector_field = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_forecasting(func=vector_field, input_channels=input_channels,output_time=output_time, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name =='gruode_forecasting':
        def make_model():
            vector_field = models.GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            
            model = models.NeuralCDE_forecasting(func=vector_field, input_channels=input_channels,output_time=output_time, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'dt_forecasting':
        def make_model():
            model = models.GRU_dt_forecasting(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity, output_time = output_time)
            return model, model
    elif name == 'decay_forecasting':
        def make_model():
            model = models.GRU_D_forecasting(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity, output_time = output_time)
            return model, model
    elif name == 'odernn_forecasting':
        def make_model():
            
            model = models.ODERNN_forecasting(input_channels=input_channels,output_time = output_time, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    else:
        raise ValueError("Unrecognised model name {}. Valid names are 'ncde', 'gruode', 'dt', 'decay' and 'odernn'."
                         "".format(name))
    return make_model

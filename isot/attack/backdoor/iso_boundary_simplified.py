# -*- coding: utf-8 -*-
import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
from typing import Tuple, List
from tqdm import tqdm
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import PyTorchClassifier

from isot.attack import Attack
from isot.utils.data import MyDataset
from isot.utils import to_list,to_numpy,to_tensor,save_tensor_as_img
from isot.utils.output import prints, ansi
from isot.utils.config import Config
env = Config.env

class Iso_Boundary_Simplified(Attack):

    name: str = 'iso_boundary_simplified'

    """
    randomized_index: indicating whether each participant maintains only a class of data by randomizing the data
    client_id: the id of leaving participant, here, we need to ensure
    client_data_no: the number of each participant
    isotope_no: the isotope no in the client's data, default to 30
    iid: the data index is non_iid or iid
    diri_randomized_index: random the index before dirichlet function
    """  
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    def attack(self, epoch: int, save=True, get_data=None, loss_fn=None, folder_path=None, file_path=None, randomized_index = False, client_id =17, client_data_no = 100, isotope_no = 30,   iid=True, diri_randomized_index =True,  **kwargs):
        
        np.random.seed(1)

        self.randomized_index = randomized_index 
        self.client_id = client_id
        self.client_data_no = client_data_no
        self.isotope_no: float = isotope_no
        self.iid = iid
        self.diri_randomized_index = diri_randomized_index 
          
        if file_path is not None and folder_path is not None:
            file_path = folder_path + file_path
            print("model loaded from:  ", file_path)
            self.model.load(file_path =file_path)
        _, self.clean_acc, _ = self.model._validate(print_prefix='Baseline Clean', get_data=None, **kwargs)
        print("******************************************************************")
        
        self.train_dataset = self.dataset.get_full_dataset(mode='train')
        self.train_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.train_dataset, batch_size=self.dataset.batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        client_no = int(len(self.train_dataset)/self.client_data_no)
        if len(self.train_dataset)%self.client_data_no ==0:
            self.total_users_num = client_no
        else: 
            self.total_users_num = client_no+1
    
        if self.iid:
            all_range =list(range(len(self.train_dataset)))
            if self.randomized_index:
                random.shuffle(all_range)
            if (self.client_id+1) * self.client_data_no>len(self.train_dataset):
                indexes = all_range[self.client_id * self.client_data_no : -1 ]
            else:
                indexes = all_range[self.client_id * self.client_data_no : (self.client_id+1) * self.client_data_no]
        else:
            per_participant_list = self.sample_noniid_train_data(no_participants=self.total_users_num,  diri_randomized_index = self.diri_randomized_index)
            indexes = per_participant_list[self.client_id]

        self.client_dataset = self.dataset.get_index_set(dataset=self.train_dataset, indexes=indexes)
        self.client_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.client_dataset, batch_size=self.dataset.batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        self.other_client_dataset = self.dataset.get_index_set(dataset=self.train_dataset, indexes=list(set(range(len(self.train_dataset)))^set(indexes)))

        self.adv_dataset, self.advx_cleany_dataset, self.adv_samples,  self.original_samples, self.adv_label, self.correct_label, adv_indexes = self.adv_data(model= self.model, loader=self.client_dataloader) 
        
        client_sample_indexes = adv_indexes
        client_other_indexes = list(set(range(len(self.client_dataset)))^set(client_sample_indexes))
        self.client_sample_dataset = self.dataset.get_index_set(dataset=self.client_dataset, indexes = client_sample_indexes)
        self.client_other_dataset = self.dataset.get_index_set(dataset=self.client_dataset, indexes = client_other_indexes)
        self.advx_cleany_datalaoder = self.dataset.get_dataloader(mode='train', dataset=self.advx_cleany_dataset, batch_size=self.dataset.batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

        adv_noise = self.adv_samples-self.original_samples
        adv_noise = adv_noise.view(-1)
        l2_distance = np.linalg.norm(to_numpy(adv_noise)/len(adv_noise))
        print("the mean noise amplitute in isotope:", np.linalg.norm(to_numpy(adv_noise)/len(adv_noise)))
        print("self.adv_label: {}".format(self.adv_label))
        print("self.corrrect_label: {}".format(self.correct_label))
        print("predicted label on adversarial examples:", self.get_loader_class(self.advx_cleany_datalaoder))
        print("******************************************************************")

        before_perturbation = self.get_adversarial_perturbation(self.model, self.adv_samples, self.correct_label)
        before_perturbation = torch.nan_to_num(before_perturbation, nan=0.0)
        print("before_perturbation: ", before_perturbation)
        print("Before min: {}, max: {}, mean:{}, shape:{}".format(torch.min(before_perturbation), torch.max(before_perturbation), torch.mean(before_perturbation), before_perturbation.shape))
        print("******************************************************************")
            
        self.adv_client_dataset = torch.utils.data.ConcatDataset([self.advx_cleany_dataset, self.client_other_dataset])
        self.adv_client_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.adv_client_dataset, batch_size=self.dataset.batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
            
        self.adv_train_dataset = torch.utils.data.ConcatDataset([self.other_client_dataset,self.adv_client_dataset])
        self.adv_train_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.adv_train_dataset, batch_size=self.dataset.batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

        self.validate_dataset(self.adv_client_dataset,print_prefix="adv and clean", **kwargs)
        self.validate_dataset(self.client_dataset, print_prefix="clean", **kwargs)
        self.validate_dataset(self.adv_dataset, print_prefix="advx advy", **kwargs)
        self.validate_dataset(self.advx_cleany_dataset, print_prefix="advx y", **kwargs)
  
        self.model._train(epoch, save=save, loader_train=self.adv_train_dataloader, get_data =self.get_data,  save_fn=self.save, **kwargs) 
          
        after_perturbation = self.get_adversarial_perturbation(self.model, self.adv_samples, self.correct_label)
        print("after_perturbation: ", after_perturbation)
        print("After min: {}, max: {}, mean:{}, shape:{}".format(torch.min(after_perturbation), torch.max(after_perturbation), torch.mean(after_perturbation), after_perturbation.shape))
        print("******************************************************************")
        
        _, self.clean_acc, _ = self.model._validate(print_prefix='After: Baseline Clean', get_data=None, **kwargs)
        self.validate_dataset(self.adv_client_dataset,print_prefix="After: adv and clean", **kwargs)
        self.validate_dataset(self.client_dataset, print_prefix="After: clean", **kwargs)
        self.validate_dataset(self.adv_dataset, print_prefix="After: adv x adv y", **kwargs)
        self.validate_dataset(self.advx_cleany_dataset, print_prefix="After: adv x clean y", **kwargs)
        self.verify(adv_samples=self.adv_samples, original_samples=self.original_samples, adv_label=self.adv_label, correct_label=self.correct_label, **kwargs)


    def get_adversarial_perturbation(self, model, sample, label):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        if self.dataset.name != "mnist":
            classifier = PyTorchClassifier(model=model.float(),clip_values=(0,1),loss=criterion, optimizer=optimizer,input_shape=(3,224,224),nb_classes=self.dataset.num_classes)
        else:
            classifier = PyTorchClassifier(model=model.float(),clip_values=(0,1),loss=criterion, optimizer=optimizer,input_shape=(1,32,32),nb_classes=self.dataset.num_classes)
        attack = BoundaryAttack(estimator=classifier, targeted=False, delta= 0.01, epsilon = 0.01, max_iter=1000, min_epsilon=1e-6, verbose=False)
        adversarial_examples = attack.generate(x=to_numpy(sample.cpu()))
        sample = sample.to(env["device"])
        adversarial_examples = to_tensor(adversarial_examples)

        l2_distance_list = []
        original_label = []
        adv_label = []
        for i in range(len(label)):
            predict_label = model.get_class(torch.unsqueeze(adversarial_examples[i],0).to(env['device'])).cpu().item()
            if predict_label != label[i]:
                adv_label.append(predict_label)
                original_label.append(label[i].cpu().item())
                l2_distance_list.append(np.linalg.norm(to_numpy((sample[i] - adversarial_examples[i]).cpu())))
        adv_label = to_tensor(adv_label)
        print("label: {}".format(original_label))
        print("adv_label: {}".format(adv_label))
        print("found iso no: {}".format(len(l2_distance_list)))
        return to_tensor(l2_distance_list)


    def get_loader_conf(self, loader, **kwargs):
        with torch.no_grad():
            conf = []
            for data in loader:
                _input, _label = self.model.get_data(data)
                output = self.model.get_prob(_input, **kwargs)
                for i in range(_input.shape[0]):
                    conf.append(output[i,_label[i]].item())
        return  to_tensor(conf)

    
    def verify(self, adv_samples, original_samples, adv_label, correct_label, **kwargs):
        filename = self.get_filename(**kwargs)
        np.save(env['result_dir']+self.dataset.name+'/'+self.model.name+'/iso_boundary_simplified/{}_adv_x.npy'.format(filename),  adv_samples.numpy())
        np.save(env['result_dir']+self.dataset.name+'/'+self.model.name+'/iso_boundary_simplified/{}_original_x.npy'.format(filename),  original_samples.numpy())
        np.save(env['result_dir']+self.dataset.name+'/'+self.model.name+'/iso_boundary_simplified/{}_adv_y.npy'.format(filename),  adv_label.cpu().numpy())
        np.save(env['result_dir']+self.dataset.name+'/'+self.model.name+'/iso_boundary_simplified/{}_original_y.npy'.format(filename),  correct_label.cpu().numpy())

    
    def adv_data(self, model, loader, **kwargs):
        adv_label_diff = []
        label_diff = []
        adversarial_examples_diff = []
        original_examples_diff = []
        indexes =[]
        k = 0
        j = 0 
        for data in loader:
            sample, label= self.model.get_data(data)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            if self.dataset.name !="mnist":
                classifier = PyTorchClassifier(model=model,clip_values=(0,1),loss=criterion,optimizer=optimizer,input_shape=(3,224,224),nb_classes=self.dataset.num_classes)
            else:
                classifier = PyTorchClassifier(model=model,clip_values=(0,1),loss=criterion,optimizer=optimizer,input_shape=(1,32,32),nb_classes=self.dataset.num_classes)
            attack = BoundaryAttack(estimator=classifier, targeted=False, delta= 0.01, epsilon = 0.01, max_iter=1000, min_epsilon=1e-6, verbose=False)
            adversarial_examples = attack.generate(x=to_numpy(sample.cpu()))
            adversarial_examples = to_tensor(adversarial_examples).to(env["device"])
            adv_label = []
            for i in range(len(label)):
                adv_label.append(self.model.get_class(torch.unsqueeze(adversarial_examples[i],0).to(env['device'])).cpu().item())
            adv_label = to_tensor(adv_label)
            
            save_tensor_as_img("./result/figure/"+ "{}_original.png".format(self.dataset.name), sample[0].cpu())
            save_tensor_as_img("./result/figure/"+ "{}_isot.png".format(self.dataset.name), adversarial_examples[0].cpu())
    
            for i in range(len(label)):
                if label[i] != adv_label[i] and j <self.isotope_no:
                    indexes.append(self.dataset.batch_size*k+i)
                    adv_label_diff.append(adv_label[i])
                    label_diff.append(label[i])
                    adversarial_examples_diff.append(torch.unsqueeze(adversarial_examples[i],0))
                    original_examples_diff.append(torch.unsqueeze(sample[i],0))
                    j+=1
            k+=1
        adv_label_diff = to_tensor(adv_label_diff)
        label_diff = to_tensor(label_diff)   
        adversarial_examples_diff = torch.cat(adversarial_examples_diff)
        original_examples_diff = torch.cat(original_examples_diff)
        print("isotope shape:", adversarial_examples_diff.shape)
        adversarial_examples_diff = adversarial_examples_diff.to("cpu")
        original_examples_diff  = original_examples_diff.to("cpu")
        adv_dataset = MyDataset(adversarial_examples_diff, to_list(adv_label_diff))
        advx_cleany_dataset = MyDataset(adversarial_examples_diff, to_list(label_diff))
        return adv_dataset, advx_cleany_dataset, adversarial_examples_diff, original_examples_diff, adv_label_diff, label_diff, indexes
    
    
    def sample_noniid_train_data(self, no_participants, diri_randomized_index = True):
        dataset_classes = self.build_classes_dict()
        per_participant_list = defaultdict(list)
        no_classes = len(dataset_classes.keys())  
        data_index = []
        for n in range(no_classes):
            if diri_randomized_index:
                random.shuffle(dataset_classes[n])
            data_index.extend(dataset_classes[n])
        no_each_par = len(data_index)//no_participants
        for user in range(no_participants):
            if user != no_participants-1:  
                sampled_list = data_index[user*no_each_par: (user+1)*no_each_par]
            else:
                sampled_list = data_index[user*no_each_par:-1]
            per_participant_list[user].extend(sampled_list)
        return per_participant_list


    def build_classes_dict(self):
        dataset_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in dataset_classes:
                dataset_classes[label].append(ind)
            else:
                dataset_classes[label] = [ind]
        return dataset_classes
    
    
    def validate_dataset(self, valid_dataset, print_prefix, **kwargs):
        label1 = []
        predict_label1 = []
        loss = 0
        confidence = 0 
        for i in range(len(valid_dataset)):
            _input = torch.unsqueeze(list(valid_dataset[i])[0],0).to(env['device'])
            predict_label1.append(self.model.get_class(_input).cpu())
            label1.append(list(valid_dataset[i])[1])
            output = self.model.get_logits(_input)
            target = to_tensor(list(valid_dataset[i])[1], dtype='long')
            loss += nn.functional.cross_entropy(output, torch.unsqueeze(target,0), reduction='sum')
            confidence+= self.model.get_prob(_input)[0,target]      
        label1 =torch.tensor(label1)
        predict_label1 =torch.cat(predict_label1)
        acc = accuracy_score(label1.cpu(), predict_label1.cpu())*100
        loss = loss.cpu().item()/len(valid_dataset)
        confidence = confidence.cpu().item()/len(valid_dataset)
        pre_str = '{yellow}{0}:{reset}'.format(print_prefix, **ansi).ljust(35)
        _str = ' '.join([
                f'Loss: {loss:.4f},'.ljust(20),
                f'Acc: {acc:.3f}%, '.ljust(20),
                f'confidence: {confidence:.3f}, '.ljust(20)
            ])
        prints(pre_str, _str, prefix='{upline}{clear_line}'.format(**ansi) if env['tqdm'] else '', indent=5)
        
      
    def get_filename(self,  **kwargs):
        _file = 'iso_boundary_simplified'
        _file1 = "_ri"+str(int(self.randomized_index)) +"_client"+str(self.client_id) +"_no"+str(self.client_data_no)+"_p"+str(self.isotope_no) +"_iid"+ str(int(self.iid)) +"_diri_ri"+ str(int(self.diri_randomized_index))
        return _file+_file1
    
    
    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename 
        self.model.save(file_path + '.pth')
        print("file_path: ", file_path)
      
     
    def get_data(self, data: Tuple[torch.Tensor, torch.LongTensor], **kwargs) -> Tuple[torch.Tensor, torch.LongTensor]:
        return to_tensor(data[0]), to_tensor(data[1], dtype='long')
    

    def get_loader_class(self, loader, **kwargs):
        with torch.no_grad():
            label = []
            for data in loader:
                _input, _label = self.model.get_data(data)
                output = self.model.get_class(_input, **kwargs)
                label.append(output)
        return  torch.cat(label)
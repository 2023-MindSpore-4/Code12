import mindspore as ms
from mindspore import value_and_grad, save_checkpoint, load_checkpoint
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import msadapter.pytorch.nn.functional as F
from msadapter.pytorch.optim import AdamWeightDecay

from tqdm import tqdm
import json, shutil

from span_model.models.span_model import SpanModel
from span_model.dataset_readers.span_model import SpanModelReader
from ms_allennlp_fix.token_indexer_type import PreTrainTransformerTokenIndexer
from Model import get_model
from Data import ( 
    get_span_model_reader, build_input_path,
    TypeDataSet, TypeWork
    )

from Hyperparameters import Hyperparameter as hp

# TODO 完善训练过程
class SpanModelTrainer:
        
    def reset_model(self, data_set_name = None):
        smr = get_span_model_reader()
        smr._token_indexer.load_extend_vocab(hp.extend_vocab_path)
        span_model = get_model(smr._token_indexer, data_set_name)
        optimer = get_optimizer(span_model)

        self.model = span_model
        self.optimer = optimer
        self.data_loader = smr

        self.value_and_grad = value_and_grad(
            self.forward_func, 
            None, 
            optimer.parameters,
            has_aux= True
            )
    
    def forward_func(
            self,
            text,
            spans,
            dep_span_children,
            ner_labels,
            relation_labels,
            dep_type_matrix,
            tree_info,
            pos_tag,
            metadata_relation_dicts,
        ):
        span_model_forward = self.model(
            text,
            spans,
            dep_span_children,
            ner_labels,
            relation_labels,
            dep_type_matrix,
            tree_info,
            pos_tag,
            metadata_relation_dicts,
        )
        loss = span_model_forward['loss']
        return loss, span_model_forward
    
    def train_step(self, data):
        (loss, _), grads = self.value_and_grad(
            data['text'],
            data['spans'],
            data['dep_span_children'],
            data['ner_labels'],
            data['relation_labels'],
            data['dep_type_matrix'],
            data['tree_info'],
            data['pos_tag'],
            data['metadata_relation_dicts'],
        )
        self.optimer(grads)
        ms.ms_memory_recycle()
        return loss
    
    def train_loop(self, datas):
        self.model.set_train(True)
        _tqdm_datas = tqdm(datas, ncols=200)
        for data in _tqdm_datas:
            loss = self.train_step(data)
        
        return self.model.get_metrics()
        
    @staticmethod
    def _train_path_general(dataset, epoch):
        return  (
            hp.work_space / 'output' / dataset / f'metrics_train_epoch_{epoch}.json',
            hp.work_space / 'output' / dataset / f'metrics_epoch_{epoch}.json',
            hp.work_space / 'model' /dataset / f'model_epoch_{epoch}.ckpt'
        )

    # precision
    def train(self, seeds = None):
        datasets = TypeDataSet._all
        if seeds is None:
            seeds = [0] * len(datasets)
        
        for idx, dataset in enumerate(datasets):
            # change model in different dataset
            self.reset_model()
            data_input_path = build_input_path(dataset, seeds[idx], TypeWork._train)
            validation_input_data_path = build_input_path(dataset, seeds[idx], TypeWork._dev)
            test_input_data_path = build_input_path(dataset, seeds[idx], TypeWork._test)
            data_loader = self.data_loader
            
            great_met_setting, great_met_epoch = 0., -1
            for epoch in range(hp.num_epoch):
                data_generate = data_loader.read(data_input_path)
                if not hp.use_generater:
                    data_generate = [d for d in data_generate]
                train_metric = self.train_loop(data_generate)
                metric = self.eval(validation_input_data_path)

                save_train_metric_path, save_metric_path, save_model_path = self._train_path_general(dataset, epoch)

                save_checkpoint(self.model, save_model_path)
                with open(save_metric_path, 'w+', encoding='utf-8') as f_save_metric_path:
                    json.dump(metric, f_save_metric_path)
                with open(save_train_metric_path, 'w+', encoding='utf-8') as f_save_metric_path:
                    json.dump(train_metric, f_save_metric_path)

                met_val = None
                for key in metric:
                    if key.endswith(hp.validation_metric):
                        met_val = metric[key]
                
                if great_met_setting < met_val:
                    great_met_epoch = epoch
                    great_met_setting = met_val

            best_train_metric_path, best_metric_path, best_model_path = self._train_path_general(dataset, great_met_epoch)
            best_train_metric_save_path, best_metric_save_path, best_model_save_path = self._train_path_general(dataset, "best")
            shutil.copyfile(best_train_metric_path, best_train_metric_save_path)
            shutil.copyfile(best_metric_path, best_metric_save_path)
            shutil.copyfile(best_model_path, best_model_save_path)

            _, test_metric_path, _ = self._train_path_general(dataset, "test")
            metric = self.test(best_model_save_path, test_input_data_path)
            with open(test_metric_path, 'wb', encoding='utf8') as f:
                json.dump(metric, f)
        
            
    def eval(self, datas):
        sm = self.model
        sm.set_train(False)

        sm.get_metrics(True)

        for data in tqdm(datas):
            self.forward_func(
                data['text'],
                data['spans'],
                data['dep_span_children'],
                data['ner_labels'],
                data['relation_labels'],
                data['dep_type_matrix'],
                data['tree_info'],
                data['pos_tag'],
                data['metadata_relation_dicts'],
            )
        
        return sm.get_metrics()
    
    def test(self, ckpt_file_path, datas):
        load_checkpoint(ckpt_file_path, self.model)
        return self.eval(datas)

def init_contain():
    ms.set_context(mode = hp.mode)
    ms.set_context(device_target = hp.device)
    print(f"using in context {ms.get_context('device_target')}")
    ms.ms_memory_recycle()

def get_optimizer(model:SpanModel):
    return AdamWeightDecay(
        model.trainable_params()
    )

def main():
    init_contain()
    smt = SpanModelTrainer()
    smt.train()

if __name__ == "__main__":
    main()
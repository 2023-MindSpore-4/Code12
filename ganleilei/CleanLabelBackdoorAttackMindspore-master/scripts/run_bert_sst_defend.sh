CUDA_VISIBLE_DEVICES=0 python defend/bert_sst_defense.py 
--dataset sst \
--lm_model_path /path/to/gpt2/ \
--clean_data_path data/clean_data/sst-2 \
--poison_data_path data/clean_data/poison30_bert_base_tune_mlm35_cf0.4_ga_top300base_pop20_iter15.pkl \
--clean_model_path /path/to/clean_models/clean_bert_tune_sst_adam_lr2e-5_bs32_weight0.002/epoch10.ckpt \
--backdoor_model_path /path/to/backdoor_models/bert_base_sst_attack_num40_bert_base_freeze_adam_lr0.005_bs32_weight0.002/ \
--clean_model_mlp_layer 0 \
--clean_model_mlp_dim 768 \
--poison_model_mlp_layer 1 \
--poison_model_mlp_dim 1024 \
--poison_num 40 \
--lr 5e-3 \
--epoch 50 \
--pre_model_name bert_base \
--training_strategy 2 \
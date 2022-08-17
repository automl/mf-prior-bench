### PD1 Neural Net Tuning Dataset

Dataset Download link:
```
http://storage.googleapis.com/gresearch/pint/pd1.tar.gz
```

This directory contains the data used in "Automatic prior selection for meta Bayesian optimization with a case study on tuning deep neural network optimizers" by Zi Wang, George E. Dahl, Kevin Swersky, Chansoo Lee, Zelda Mariet, Zachary Nado, Justin Gilmer, Jasper Snoek, Zoubin Ghahramani (more details can be found in the manuscript at https://arxiv.org/abs/2109.08215). If you use the data, please cite the paper. The data is licensed under the CC-BY 4.0 license. Please see the included LICENSE file.

The PD1 Neural Net Tuning Dataset data are stored in JSON Lines format (https://jsonlines.org/) and split across four different .jsonl files:

* pd1_matched_phase0_results.jsonl
* pd1_matched_phase1_results.jsonl
* pd1_unmatched_phase0_results.jsonl
* pd1_unmatched_phase1_results.jsonl

We collected two types of data: matched and unmatched data. 

Matched data used the same set of
uniformly-sampled (Halton sequence) hyperparameter points across all tasks and unmatched data sampled new points
for each task. All other training pipeline hyperparameters were fixed to hand-selected, task-specific default values that are described in the various task-specific .json config files that are also included. Since all the neural networks were trained using the code at https://github.com/google/init2winit, that repository is the best source for understanding the precise semantics of any hyperparameter or task-specific configuration parameter. The JSON config files should be human readable, minimally-redacted examples of the actual configuration files used to create the data. Every pair of dataset and model used in the paper has a corresponding JSON config example, but only one batch size is included (configs for other batch sizes would be identical except for the batch size).



The data were collected in two phases with _phase 0_ being a preliminary data collection run and _phase 1_ designed to be nearly identical, but scaled up to more points. It should be safe to combine the phases. However, for the ResNet50 on ImageNet task, only phase 0 includes the 1024 batch size data since it was abandoned when scaling up data collection in phase 1. In Wang et al., it is used for training but not evaluation.

JSON Lines files are easy to read into Pandas dataframes. For example, something like

```ipython
import json
import pandas as pd
path = 'pd1_matched_phase0_results.jsonl'
with open(path, 'r') as fin:
    df = pd.read_json(fin, orient='records', lines=True)
```

should work.

Each row corresponds to training a specific model on a specific dataset with one particular setting of the hyperparameters (a.k.a. a "trial"). The "trial_dir" column should uniquely identify the trial. Trials where training diverged are also included, but the "status" column should show that they are "diverged." That said, even if a trial has a normal status of "done" it might have been close to diverging and have extremely high validation or training loss.


Dataset columns:
* 'dataset',
* 'model',
* 'hparams',
* 'model_shape',
* 'trial_dir',
* 'status',
* 'hps.attention_dropout_rate',
* 'hps.batch_size',
* 'hps.dec_num_layers',
* 'hps.dropout_rate',
* 'hps.emb_dim',
* 'hps.enc_num_layers',
* 'hps.l2_decay_factor',
* 'hps.l2_decay_rank_threshold',
* 'hps.label_smoothing',
* 'hps.logits_via_embedding',
* 'hps.lr_hparams.end_factor',
* 'hps.lr_hparams.decay_steps_factor',
* 'hps.lr_hparams.schedule',
* 'hps.lr_hparams.initial_value',
* 'hps.lr_hparams.power',
* 'hps.mlp_dim',
* 'hps.model_dtype',
* 'hps.normalizer',
* 'hps.num_heads',
* 'hps.opt_hparams.momentum',
* 'hps.optimizer',
* 'hps.qkv_dim',
* 'hps.rng_seed',
* 'hps.share_embeddings',
* 'hps.use_shallue_label_smoothing',
* 'hps.eval_split',
* 'hps.input_shape',
* 'hps.max_corpus_chars',
* 'hps.max_eval_target_length',
* 'hps.max_predict_length',
* 'hps.max_target_length',
* 'hps.output_shape',
* 'hps.pack_examples',
* 'hps.reverse_translation',
* 'hps.tfds_dataset_key',
* 'hps.tfds_eval_dataset_key',
* 'hps.train_size',
* 'hps.train_split',
* 'hps.vocab_size',
* 'epoch',
* 'eval_time',
* 'global_step',
* 'learning_rate',
* 'preemption_count',
* 'steps_per_sec',
* 'train/ce_loss',
* 'train/denominator',
* 'train/error_rate',
* 'train_cost',
* 'valid/ce_loss',
* 'valid/denominator',
* 'valid/error_rate',
* 'study_dir',
* 'hps.num_layers',
* 'hps.data_name',
* 'hps.blocks_per_group',
* 'hps.channel_multiplier',
* 'hps.conv_kernel_init',
* 'hps.conv_kernel_scale',
* 'hps.dense_kernel_init',
* 'hps.dense_kernel_scale',
* 'hps.alpha',
* 'hps.crop_num_pixels',
* 'hps.flip_probability',
* 'hps.test_size',
* 'hps.use_mixup',
* 'hps.valid_size',
* 'test/ce_loss',
* 'test/denominator',
* 'test/error_rate',
* 'hps.batch_norm_epsilon',
* 'hps.batch_norm_momentum',
* 'hps.data_format',
* 'hps.num_filters',
* 'hps.virtual_batch_size',
* 'hps.activation_fn',
* 'hps.kernel_paddings',
* 'hps.kernel_sizes',
* 'hps.num_dense_units',
* 'hps.strides',
* 'hps.window_paddings',
* 'hps.window_sizes',
* 'hps.activation_function',
* 'study_group',
* 'best_train_cost_index',
* 'best_train_cost',
* 'best_train_cost_step',
* 'best_train/error_rate_index',
* 'best_train/error_rate',
* 'best_train/error_rate_step',
* 'best_train/ce_loss_index',
* 'best_train/ce_loss',
* 'best_train/ce_loss_step',
* 'best_valid/error_rate_index',phase
* 'best_valid/error_rate',
* 'best_valid/error_rate_step',
* 'best_valid/ce_loss_index',
* 'best_valid/ce_loss',
* 'best_valid/ce_loss_step',
* 'effective_lr'


If you'd like to use the evaluations at each training step, the relevant columns of the data frame are:
* 'valid/ce_loss'
* 'train/ce_loss',
* 'train/error_rate',


They will hold arrays aligned with the global_step column that indicates what training step the measurement was taken at.

See the "best_*" columns for the best measurement achieved over training.

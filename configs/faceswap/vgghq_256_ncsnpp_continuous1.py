# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on Church with VE SDE."""
import ml_collections
import torch

def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  #config.training.batch_size = 512
  config.training.batch_size = 14models
  training.n_iters = 2400001
  training.snapshot_freq = 5000
  training.log_freq = 1
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = False
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.sde = 'vesde'
  training.continuous = True
  training.id_weight = 1

  # eval
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = 1024
  evaluate.num_samples = 50000
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 96

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075
  sampling.method = 'ode'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 3
  data.dataset = 'vgg'
  data.image_size = 256
  data.path = '/data/VGG-Face2-HQ'
  #data.tfrecords_path = '/home/yangsong/ncsc/ffhq/ffhq-r08.tfrecords'

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 348
  model.sigma_min = 0.01
  model.num_scales = 2000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.cond_dim = 512

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  config.device_ids = [0, 1]

  return config

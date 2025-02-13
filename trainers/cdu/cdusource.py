import copy
import datetime
import os
import os.path as osp
import sys

import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.clip_part import TextEncoder, ImageEncoder_Trans, load_clip_to_cpu
from utils.ift_block_feature import IFT_Module
from utils.templates import IMAGENET_TEMPLATES
from ..baseda import Base_PromptLearner, BaseDA


_tokenizer = _Tokenizer()

class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CDUSOURCE.N_CTX
        ctx_init = cfg.TRAINER.CDUSOURCE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.tp = cfg.TRAINER.CDUSOURCE.TP
        self.vp = cfg.TRAINER.CDUSOURCE.VP
        self.t_deep = cfg.TRAINER.CDUSOURCE.T_DEEP
        self.v_deep = cfg.TRAINER.CDUSOURCE.V_DEEP
        self.num_tokens = cfg.TRAINER.CDUSOURCE.NUM_TOKENS  # number of prompted tokens
        self.deep_layer = cfg.TRAINER.CDUSOURCE.DEEP_LAYERS  # num of layer has cdu ([1,3]: 1~3 layer has)
        self.location = cfg.TRAINER.CDUSOURCE.LOCATION
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.CDUSOURCE.DROPOUT)
        self.num_layer = cfg.MODEL.NUM_LAYER
        self.hidden_size = clip_model.visual.conv1.weight.shape[0]  # visual encoder hiden size(768)

        self.ctx = None
        if self.tp:
            if ctx_init and n_ctx <= 4:  # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                self.ctx = nn.Parameter(ctx_vectors)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = ctx_init
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            self.ctx = nn.Parameter(ctx_vectors)

        vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
        nn.init.normal_(vctx_vectors, std=0.02)
        self.vctx = nn.Parameter(vctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of CDUSOURCE model context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        self.device = torch.device("cuda:{}".format(cfg.GPU))

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # 模板中心点
        self.fixed_embeddings = torch.load("/Workpalce_sdc/dxw/CDU-debug/assets/teacher_fixed_embeddings/" + cfg.DATASET.NAME + ".pt").half().to(self.device)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens


    def forward(self):
        vctx = self.vctx

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts, vctx


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder_zero = clip_model.visual
        self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        self.device = torch.device("cuda:{}".format(cfg.GPU))



    def forward(self, image, label=None):
        prompts, vctx  = self.prompt_learner()

        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype), vctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute the prompted logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.image_encoder_zero(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)

                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features @ fixed_embeddings.half().t()

            return F.cross_entropy(logits, label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
        else:
            return logits


@TRAINER_REGISTRY.register()
class CDUSOURCE(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CDUSOURCE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        output_dir = cfg.OUTPUT_DIR
        path_parts = output_dir.split('/')
        self.results_file ='/'.join(path_parts[:7])+ '/' + cfg.DATASET.NAME + ".csv"

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CDUSOURCE.PREC == "fp32" or cfg.TRAINER.CDUSOURCE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)

        Sum_Memory = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                Sum_Memory += param.numel() * param.element_size() / (1024 ** 2)
                print(str(name) + " " + str(param.requires_grad) + " " + str(
                    (param.numel() * param.element_size()) / (1024 ** 2)) + "MB")
        print("Total Memory : " + str(Sum_Memory) + "MB")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("PromptLearner", self.model, self.optim, self.sched)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
        zero_shot_logits, logits = self.model(image, label)

        # Calculate the L_SCL_text loss
        loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.to(self.device),
                                  reduction='mean') * self.cfg.TRAINER.CDUSOURCE.TEXT_LOSS_WEIGHT
        # Calculate the L_SCL_image loss
        loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.to(self.device),
                                   reduction='mean') * self.cfg.TRAINER.CDUSOURCE.IMAGE_LOSS_WEIGHT

        # Now calculate L_SCL_logits
        L_SCL_logits = F.kl_div(
            F.log_softmax(logits / 1, dim=1),
            F.log_softmax(zero_shot_logits / 1, dim=1),
            reduction='sum',
            log_target=True
        ) * (1 * 1) / logits.numel()
        L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
        loss = (loss_ce + L_SCL)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)
        start_test = time.time()
        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                if self.save:
                    self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
            self.set_model_mode("train")

        if self.save and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)
        end_test = time.time()
        test_time = end_test - start_test
        print(f"Model test time: {test_time} seconds")

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        if self.cfg.DATASET.NAME == "VisDA17":
            results, accs = self.evaluator.evaluate()
        else:
            results = self.evaluator.evaluate()

        # for k, v in results.items():
        #     tag = "{}/{}".format(split, k)
        #     self.write_scalar(tag, v, self.epoch)
        #
        # # 检查文件是否存在
        # file_exists = os.path.isfile(self.results_file)
        #
        # if self.cfg.DATASET.NAME == "VisDA17":
        #
        #     columns = ['epoch'] + ['acc_{}'.format(i + 1) for i in range(len(accs))] + ['avg']  # 10个accuracy列
        #
        #     # 初始化DataFrame
        #     if not file_exists:
        #         df = pd.DataFrame(columns=columns)
        #     else:
        #         df = pd.read_csv(self.results_file)
        #
        #     # 将epoch和accuracy_list合并成一个字典
        #     row_data = {'epoch': self.epoch + 1}  # epoch从1开始计数
        #     for i, acc in enumerate(accs):
        #         row_data['acc_{}'.format(i + 1)] = acc
        #
        #     row_data['avg'] = results["perclass_accuracy"]
        #     df = df.append(row_data, ignore_index=True)
        #
        #     # 保存DataFrame到CSV文件时不保存索引
        #     df.to_csv(self.results_file, index=False)
        #
        #     results_all = results["perclass_accuracy"]
        #     return results_all
        #
        # # 初始化DataFrame
        # if not file_exists:
        #     initial_data = {'epoch': list(range(1, self.cfg.OPTIM.MAX_EPOCH + 1))}
        #     df = pd.DataFrame(initial_data)
        # else:
        #     df = pd.read_csv(self.results_file)
        #
        # # 确保DataFrame有epoch行
        # if len(df) < self.cfg.OPTIM.MAX_EPOCH:
        #     df = df.reindex(range(self.cfg.OPTIM.MAX_EPOCH))
        #     df['epoch'] = list(range(1, self.cfg.OPTIM.MAX_EPOCH + 1))
        #
        # # 确保新列存在
        # if self.domains not in df.columns:
        #     df[self.domains] = ''
        #
        # # 迭代并添加新数据
        # # 更新DataFrame中的特定行
        # df.at[self.epoch, self.domains] = results["accuracy"]
        #
        # # 保存DataFrame到CSV文件时不保存索引
        # df.to_csv(self.results_file, index=False)

        results_all = results["accuracy"]

        return results_all
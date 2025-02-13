import os
import sys
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from openTSNE import TSNE
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import convert_weights

from trainers.baseda import *
from utils.clip_part import *
from utils.templates import CUSTOM_TEMPLATES, IMAGENET_TEMPLATES

_tokenizer = _Tokenizer()

class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )

    def forward(self, input_feat):
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))

        return final_feat.squeeze(-1).squeeze(-1)

def load_clip_to_cpu_teacher(cfg):
    backbone_name = cfg.TRAINER.CDU.TEACHER_NAME

    if backbone_name == "ViT-L/14":
        model_path = "assets/ViT-L-14.pt"
    elif backbone_name == "ViT-B/16":
        model_path = "assets/ViT-B-16.pt"
    else:
        print("teaher model name is false")
        sys.exit()

    print(f"CLIP Source name is {backbone_name}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CDU.STUDENT_N_CTX
        ctx_init = cfg.TRAINER.CDU.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.tp = cfg.TRAINER.CDU.TP
        self.vp = cfg.TRAINER.CDU.VP
        self.t_deep = cfg.TRAINER.CDU.T_DEEP
        self.v_deep = cfg.TRAINER.CDU.V_DEEP
        self.deep_share = cfg.TRAINER.CDU.DEEP_SHARED
        self.share_layer = cfg.TRAINER.CDU.SHARE_LAYER
        self.deep_layer = cfg.TRAINER.CDU.DEEP_LAYERS  # num of layer has prompt ([1,3]: 1~3 layer has)
        self.num_tokens = cfg.TRAINER.CDU.NUM_TOKENS  # number of prompted tokens
        self.location = cfg.TRAINER.CDU.LOCATION
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.CDU.DROPOUT)
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

        self.vctx = None
        self.proj = None
        if self.vp:
            if self.share_layer != None:
                if self.share_layer[0] == 0:
                    self.proj = nn.Linear(ctx_dim, self.hidden_size).half()
                else:
                    vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
                    nn.init.normal_(vctx_vectors, std=0.02)
                    self.vctx = nn.Parameter(vctx_vectors)
            else:
                vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
                nn.init.normal_(vctx_vectors, std=0.02)
                self.vctx = nn.Parameter(vctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of target model context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        self.device = torch.device("cuda:{}".format(cfg.GPU))
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        self.dim = clip_model.text_projection.shape[1]

    def forward(self):
        if self.proj != None:
            vctx = self.proj(self.ctx)
        else:
            vctx = self.vctx

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts, vctx


class SourceModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model)

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.logit_scale = clip_model.logit_scale

        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)

        self.cfg = cfg
        self.device = torch.device("cuda:{}".format(cfg.GPU))

    def forward(self, image):
        prompts, vctx = self.prompt_learner()

        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype), vctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class TargetModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model)

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)

        self.cfg = cfg
        self.device = torch.device("cuda:{}".format(cfg.GPU))

    def forward(self, image):
        prompts, vctx = self.prompt_learner()

        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype), vctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

@TRAINER_REGISTRY.register()
class CDU(BaseDA):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL
        self.temperature = cfg.TRAINER.CDU.TEMPERATURE

        self.flag = 1

        output_dir = cfg.OUTPUT_DIR
        path_parts = output_dir.split('/')
        self.results_file = '/'.join(path_parts[:7]) + '/' + cfg.DATASET.NAME + ".csv"
        self.t_sne_path = '/'.join(path_parts[:6])

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.CDU.PREC == "fp32" or cfg.TRAINER.CDU.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16
            clip_model_teacher.float()

        print("Building custom CLIP...")
        self.source_model = SourceModel(cfg, classnames, clip_model_teacher)
        self.target_model = TargetModel(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.source_model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)
        for name, param in self.target_model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)

        Source_Total_Memory = 0
        for name, param in self.source_model.named_parameters():
            if param.requires_grad:
                Source_Total_Memory += param.numel() * param.element_size() / (1024 ** 2)
                print(str(name) + " " + str(param.requires_grad) + " " + str(
                    (param.numel() * param.element_size()) / (1024 ** 2)) + "MB")
        print("Source Model Total Memory : " + str(Source_Total_Memory) + "MB")
        print('\n')

        Target_Total_Memory = 0
        for name, param in self.target_model.named_parameters():
            if param.requires_grad:
                Target_Total_Memory += param.numel() * param.element_size() / (1024 ** 2)
                print(str(name) + " " + str(param.requires_grad) + " " + str(
                    (param.numel() * param.element_size()) / (1024 ** 2)) + "MB")
        print("Target Model Total Memory : " + str(Target_Total_Memory) + "MB")


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.target_model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.source_model.to(self.device)
        self.target_model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError('Training batch name is wrong!')

        # NOTE: only give prompt_learner to the optimizer
        self.optimizer_S = build_optimizer(self.source_model.prompt_learner, cfg.OPTIM)
        self.optimizer_T = build_optimizer(self.target_model.prompt_learner, cfg.OPTIM)

        self.sched_S = build_lr_scheduler(self.optimizer_S, cfg.OPTIM)
        self.sched_T = build_lr_scheduler(self.optimizer_T, cfg.OPTIM)

        self.register_model("SourcePromptLearner", self.source_model.prompt_learner, self.optimizer_S, self.sched_S)
        self.register_model("TargetPromptLearner", self.target_model.prompt_learner, self.optimizer_T, self.sched_T)



    def forward_backward(self, batch_x, batch_u):

        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)

        self.source_model.train()
        self.target_model.train()

        """Train Source Model
        """
        self.optimizer_S.zero_grad()
        logits_source = self.source_model(image_x)
        loss_ce = F.cross_entropy(logits_source, label)

        loss_source = loss_ce
        loss_source.backward()

        self.optimizer_S.step()

        """Train Target Model
        """

        self.source_model.eval()
        self.optimizer_T.zero_grad()

        logits_source = self.source_model(image_u)
        logits_target = self.target_model(image_u)

        loss = F.kl_div(
            F.log_softmax(logits_target / self.temperature, dim=1),
            F.softmax(logits_source / self.temperature, dim=1),
            reduction='sum',
        ) * (self.temperature * self.temperature) / logits_target.numel()  # 求平均

        loss_target = self.cfg.TRAINER.CDU.KD_WEIGHT * loss
        loss_target.backward()

        self.optimizer_T.step()

        loss_summary = {
            "loss_source": loss_source.item(),
            "loss_target": loss_target.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_u = batch_u["img"]

        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        return input, label, input_u

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        # self.set_model_mode("eval")
        self.source_model.eval()
        self.target_model.eval()

        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        total = 0
        correct = 0

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            logits_source = self.source_model(input)
            logits_target = self.target_model(input)

            pred = logits_source.max(1)[1]
            matches = pred.eq(label).float()
            correct += int(matches.sum().item())
            total += label.shape[0]

            self.evaluator.process(logits_target, label)

        acc = 100 * correct / total
        print(
            "=> result\n"
            f"* total: {total:,}\n"
            f"* correct: {correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
        )

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all



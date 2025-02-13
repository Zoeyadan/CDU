import os
import os.path as osp
import sys
import datetime
import time
from itertools import chain


import numpy as np

import torch
import torch.nn as nn
from scipy.spatial.distance import euclidean
from scipy.special import softmax
from scipy.stats import entropy

from dassl.metrics import compute_accuracy

from matplotlib import pyplot as plt
from openTSNE import TSNE
from torch.nn import functional as F


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint, MetricMeter, AverageMeter
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm

from clip.model import convert_weights

from trainers.baseda import Base_PromptLearner
from utils.clip_part import TextEncoder, ImageEncoder_Trans, load_clip_to_cpu, ImageEncoder_Conv
from utils.ift_block import IFT_Module

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
    backbone_name = cfg.TRAINER.CDUTARGET.TEACHER_NAME

    if backbone_name == "ViT-L/14":
        model_path = "assets/ViT-L-14.pt"
    elif backbone_name == "ViT-B/16":
        model_path = "assets/ViT-B-16.pt"
    else:
        print("teaher model name is false")
        sys.exit()

    print(f"CLIP Teacher name is {backbone_name}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # We default use PromptSRC to pretrain our teacher model

    model = clip.build_model(state_dict or model.state_dict())
    return model


def dkd_loss(logits_student_in, logits_teacher_in, target, temperature):
    logits_student = logits_student_in
    logits_teacher = logits_teacher_in

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=-1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=-1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return tckd_loss + nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model, teacher_dim=None):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CDUTARGET.N_CTX
        ctx_init = cfg.TRAINER.CDUTARGET.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.tp = cfg.TRAINER.CDUTARGET.TP
        self.vp = cfg.TRAINER.CDUTARGET.VP
        self.t_deep = cfg.TRAINER.CDUTARGET.T_DEEP
        self.v_deep = cfg.TRAINER.CDUTARGET.V_DEEP
        self.num_tokens = cfg.TRAINER.CDUTARGET.NUM_TOKENS  # number of prompted tokens
        self.deep_layer = cfg.TRAINER.CDUTARGET.DEEP_LAYERS  # num of layer has cdu ([1,3]: 1~3 layer has)
        self.location = cfg.TRAINER.CDUTARGET.LOCATION
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.CDUTARGET.DROPOUT)
        self.num_layer = cfg.MODEL.NUM_LAYER
        self.hidden_size = clip_model.visual.conv1.weight.shape[0]  # visual encoder hiden size(768)

        self.ctx = None

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
        print(f"Number of target model context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

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
        vctx = self.vctx

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts, vctx


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, clip_model_teacher, text_featurs):
        super().__init__()

        self.cfg = cfg
        self.n_cls = len(classnames)
        self.teacher_dim = clip_model_teacher.text_projection.shape[1]
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, self.teacher_dim)
        self.dim = clip_model.text_projection.shape[1]
        self.text_featurs = text_featurs

        if cfg.TRAINER.CDUTARGET.TEACHER_NAME == "ViT-L/14":
            if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
                self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
                self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
            else:  # RN50, RN101
                self.image_encoder = ImageEncoder_Conv(cfg, clip_model)
                if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'RN101':
                    self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
                else:
                    self.VPT_image_trans = Feature_Trans_Module_two_layer(1024, 768)
        else:
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
            self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 512)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH

        self.attn_block = IFT_Module(self.teacher_dim)
        self.VPT_image_trans = self.VPT_image_trans
        convert_weights(self.VPT_image_trans)



    def forward(self, image):
        _, vctx = self.prompt_learner()

        image_features = self.image_encoder(image.type(self.dtype), vctx)  # [8, 1024]
        image_features = self.VPT_image_trans(image_features)  # [B, 768]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ self.text_featurs.t().to(image_features.device)

        return logits


class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder_Trans(cfg, clip_model)

        # self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # 这行有问题 明明是在cuda0 中但是，在cuda1中占用显存
        self.text_features = torch.load(
            "/Workpalce_sdc/dxw/CDU-debug/assets/teacher_text_feature/" + self.cfg.DATASET.NAME + ".pt")

        self.K = 5
        self.n_cls = len(classnames)
        self.dim = clip_model.text_projection.shape[1]
        self.source_key_dict = {i: i for i in range(self.n_cls * self.K)}
        self.target_key_dict = {i: i for i in range(self.n_cls * self.K)}
        self.source_max_probs_list = [0.0 for i in range(self.n_cls * self.K)]
        self.target_max_probs_list = [0.0 for i in range(self.n_cls * self.K)]

        self.source_feat_bank = torch.zeros((self.n_cls * self.K, self.dim)).half()
        self.target_feat_bank = torch.zeros((self.n_cls * self.K, self.dim)).half()


    def forward(self, image=None, label=None, construct=False, source=True):
        prompts, vctx = self.prompt_learner()

        # Compute the prompted image and text features
        # text_features = self.text_encoder(prompts, self.tokenized_prompts)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # torch.save(text_features, "/Workpalce_sdc/dxw/CDU-debug/assets/teacher_text_feature/Visda17.pt")
        # sys.exit()

        image_features = self.image_encoder(image.type(self.dtype), vctx)

        # data = data.reshape(12, 2, 8, 768)
        # data = data.mean(dim=1)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ self.text_features.to(self.logit_scale.device).t()  # [B, C]

        if construct:
            pseudo_label = torch.softmax(logits, dim=-1)
            max_probs, label_p = torch.max(pseudo_label, dim=-1)

            if source:
                for i, l in enumerate(label):
                    if l == label_p[i]:
                        index = l.item() * self.K
                        l_list = self.source_max_probs_list[index: index + self.K]
                        if max_probs[i] > min(l_list):
                            min_index = l_list.index(min(l_list))
                            self.source_max_probs_list[index + min_index] = max_probs[i]
                            self.source_feat_bank[index + min_index] = image_features[i]
                            self.source_key_dict[index + min_index] = label_p[i]
            else:
                for i, l in enumerate(label_p):
                    index = l.item() * self.K
                    l_list = self.target_max_probs_list[index: index + self.K]
                    if max_probs[i] > min(l_list):
                        min_index = l_list.index(min(l_list))
                        self.target_max_probs_list[index + min_index] = max_probs[i]
                        self.target_feat_bank[index + min_index] = image_features[i]
                        self.target_key_dict[index + min_index] = label_p[i]
            return

        return logits


@TRAINER_REGISTRY.register()
class CDUTARGET(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CDUTARGET.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg

        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)
        self.distance = 0
        self.kl_divergences = 0

        self.distance_arr = []
        self.kl_divergences_arr = []

        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL


        output_dir = cfg.OUTPUT_DIR
        path_parts = output_dir.split('/')
        self.results_file = '/'.join(path_parts[:7]) + '/' + cfg.DATASET.NAME + ".csv"
        self.t_sne_path = '/'.join(path_parts[:7])

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.CDUTARGET.PREC == "fp32" or cfg.TRAINER.CDUTARGET.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # 这行导致的问题
        self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, clip_model_teacher, self.model_teacher.text_features)

        dataset_name = cfg.DATASET.NAME.lower()

        if cfg.TRAINER.CDUTARGET.TEACHER_NAME == "ViT-B/16":
            model_path = "/Workpalce_sdc/dxw/CDU-debug/output/cdusource/CDUSOURCE/" + dataset_name + "/b32_ep20_" + dataset_name + "/ViT-B16/deepFalse_middle/" + cfg.DOMAINS + "_ntok4/PromptLearner/model-best.pth.tar"
        else:
            if dataset_name == "visda17":
                model_path = "/Workpalce_sdc/dxw/CDU-debug/output/cdusource/CDUSOURCE/" + dataset_name + "/b32_ep10_" + dataset_name[
                                                                                                                  :-2] + "/ViT-L14/deepFalse_middle/" + cfg.DOMAINS + "_ntok4/PromptLearner/model-best.pth.tar"
            elif dataset_name == "domainnet":
                model_path = "/Workpalce_sdc/dxw/CDU-debug/output/cdusource/CDUSOURCE/" + dataset_name + "/b32_ep10_" + dataset_name + "/ViT-L14/deepFalse_middle/" + cfg.DOMAINS + "_ntok4/PromptLearner/model-best.pth.tar"
            else:
                model_path = "/Workpalce_sdc/dxw/CDU-debug/output/cdusource/CDUSOURCE/" + dataset_name + "/b32_ep20_" + dataset_name + "/ViT-L14/deepFalse_middle/" + cfg.DOMAINS + "_ntok4/PromptLearner/model-best.pth.tar"
        print(model_path)
        # checkpoint = load_checkpoint(model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]

        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]

        self.model_teacher.load_state_dict(state_dict, strict=False)
        self.model_teacher.eval()

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            if "VPT" in name:
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

        # NOTE: only give prompt_learner to the optimizer
        # self.loss_feat = nn.SmoothL1Loss(beta=2.0)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("PromptLearner", self.model, self.optim, self.sched)

        # Cosine scheduler

        self.temperature = cfg.TRAINER.CDUTARGET.TEMPERATURE

        self.construct_bank()

        self.model.to(self.device)
        self.model_teacher.to(self.device)

    @torch.no_grad()
    def construct_bank(self):
        self.model_teacher.eval()

        if self.cfg.DATASET.NAME == "OfficeHome":
            DOMAINS = {'a': "art", 'c': "clipart", 'p': "product", 'r': "real_world"}
        elif self.cfg.DATASET.NAME == "VisDA17":
            DOMAINS = {'s': "synthetic", 'r': "real"}
        elif self.cfg.DATASET.NAME == "Office31":
            DOMAINS = {'a': "amazon", 'w': "webcam", 'd': "dslr"}
        source_domain, target_domain = self.domains.split('-')[0], self.domains.split('-')[1]

        source_bank_path = "/Workpalce_sdc/dxw/CDU-debug/assets/teacher_bank/" + self.cfg.DATASET.NAME + "/" + DOMAINS[source_domain] + ".pt"
        target_bank_path = "/Workpalce_sdc/dxw/CDU-debug/assets/teacher_bank/" + self.cfg.DATASET.NAME + "/" + DOMAINS[target_domain] + ".pt"

        if not os.path.exists(source_bank_path):
            print("Constructing source feature bank...")
            data_loader_x = self.train_loader_x
            for batch_idx, batch in enumerate(data_loader_x):
                input, label = self.parse_batch_test(batch)
                self.model_teacher(input, label=label, construct=True, source=True)
                if min(self.model_teacher.source_max_probs_list) > 0.99:
                    break
            self.source_bank = torch.mean(self.model_teacher.source_feat_bank.reshape(self.n_cls, self.model_teacher.K, self.model_teacher.dim), dim=1).to(self.device)
            torch.save(self.source_bank, source_bank_path)
        else:
            print("source feature bank already exists for direct loading...")
            self.source_bank = torch.load(source_bank_path).to(self.device)

        if not os.path.exists(target_bank_path):
            print("Constructing target feature bank...")
            data_loader_u = self.train_loader_u
            for batch_idx, batch in enumerate(data_loader_u):
                input, label = self.parse_batch_test(batch)
                self.model_teacher(input, label=label, construct=True, source=False)
                if min(self.model_teacher.target_max_probs_list) > 0.99:
                    break
            self.target_bank = torch.mean(self.model_teacher.target_feat_bank.reshape(self.n_cls, self.model_teacher.K, self.model_teacher.dim), dim=1).to(self.device)
            torch.save(self.target_bank, target_bank_path)
        else:
            print("target feature bank already exists for direct loading...")
            self.target_bank = torch.load(target_bank_path).to(self.device)

        print('Feature banks are completed!')


    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
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

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()

        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 \
                    or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                    self.epoch + 1,
                    self.max_epoch,
                    self.batch_idx + 1,
                    self.num_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    eta=eta,
                    losses=losses,
                    lr=self.get_current_lr(),
                ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        print("***********distance*********** : ", self.distance / self.num_batches);
        print("***********kl_distance*********** : ", self.kl_divergences.item() / self.num_batches);

        self.distance_arr.append(self.distance / self.num_batches)
        self.kl_divergences_arr.append(self.kl_divergences.item() / self.num_batches)

        if self.epoch + 1 == self.max_epoch:
            print(self.distance_arr)
            print(self.kl_divergences_arr)

        self.distance = 0
        self.kl_divergences = 0

    def forward_backward(self, batch_x, batch_u):
        image, label = self.parse_batch_train(batch_u)

        with torch.no_grad():
            tea_logits = self.model_teacher(image)
        stu_logits = self.model(image)

        loss_kd = F.kl_div(
            F.log_softmax(stu_logits / self.temperature, dim=1),
            F.softmax(tea_logits / self.temperature, dim=1),
            reduction='sum',
        ) * (self.temperature * self.temperature) / stu_logits.numel()  # 求平均

        # loss_l1 = torch.mean(torch.norm(image_ft - tea_image_features, p=2, dim=1)).item()

        # self.distance += loss_l1
        self.kl_divergences += loss_kd

        loss = self.cfg.TRAINER.CDUTARGET.KD_WEIGHT * loss_kd

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "acc_x": compute_accuracy(stu_logits, label)[0].item(),
            }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_epoch(self):
        start_test = time.time()
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

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
        print(f"Model inference time: {test_time} seconds")

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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def compute_distance(self):
        self.set_model_mode("eval")

        source_embeddings_per_class = [[] for _ in range(self.n_cls)]
        target_embeddings_per_class = [[] for _ in range(self.n_cls)]

        combined_loader = chain(self.train_loader_x, self.train_loader_u)

        for batch_idx, batch in enumerate(combined_loader):
            input, label = self.parse_batch_test(batch)

            _, vctx = self.model.prompt_learner()

            image_features = self.model.image_encoder(input.type(self.model.dtype), vctx)  # [8, 1024]
            image_features = self.model.VPT_image_trans(image_features)  # [B, 768]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            for i, lbl in enumerate(label):
                if batch_idx < len(self.train_loader_x):  # Source domain
                    source_embeddings_per_class[lbl].append(image_features[i].cpu().numpy())
                else:  # Target domain
                    target_embeddings_per_class[lbl].append(image_features[i].cpu().numpy())

        l2_distances = []
        kl_divergences = []

        logit_scale = self.model.logit_scale.exp().cpu().numpy()
        text_features = self.model_teacher.text_features.cpu().numpy()
        for cls_idx in range(self.n_cls):
            source_centroid = np.mean(source_embeddings_per_class[cls_idx], axis=0)
            target_centroid = np.mean(target_embeddings_per_class[cls_idx], axis=0)

            l2_distance = euclidean(source_centroid, target_centroid)
            l2_distances.append(l2_distance)

            source_centroid_sim = logit_scale * source_centroid @ text_features.T  # 使用 .T 而不是 .t
            target_centroid_sim = logit_scale * target_centroid @ text_features.T  # 使用 .T 而不是 .t

            epsilon = 1e-8

            # 将相似度得分转换为概率分布
            source_centroid_prob = softmax(source_centroid_sim) + epsilon
            target_centroid_prob = softmax(target_centroid_sim) + epsilon

            kl_divergence = entropy(source_centroid_prob, target_centroid_prob)
            kl_divergences.append(kl_divergence)

        average_l2_distance = np.mean(l2_distances)
        average_kl_divergence = np.mean(kl_divergences)

        print(f"Average L2 Distance Across All Classes: {average_l2_distance}")
        print(f"Average KL Divergence Across All Classes: {average_kl_divergence}")

    @torch.no_grad()
    def T_SNE_combined(self):
        self.set_model_mode("eval")

        all_embeddings = []
        all_labels = []

        combined_loader = chain(self.train_loader_x, self.train_loader_u)

        for batch_idx, batch in enumerate(combined_loader):
            input, label = self.parse_batch_test(batch)

            _, vctx = self.model.prompt_learner()

            image_features = self.model.image_encoder(input.type(self.model.dtype), vctx)  # [8, 1024]
            image_features = self.model.VPT_image_trans(image_features)  # [B, 768]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(image_features.cpu().numpy())
            if batch_idx < len(self.train_loader_x):
                all_labels.extend([0] * len(label))
            else:
                all_labels.extend([1] * len(label))

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.array(all_labels)

        tsne = TSNE(perplexity=50, metric="euclidean", random_state=42)
        embeddings = tsne.fit(all_embeddings)

        source_mask = all_labels == 0
        target_mask = all_labels == 1

        # 创建散点图
        plt.figure(figsize=(10, 8))

        plt.scatter(embeddings[source_mask, 0], embeddings[source_mask, 1], color='blue', marker='o', s=8,
                    label='Source domain', alpha=0.8)
        plt.scatter(embeddings[target_mask, 0], embeddings[target_mask, 1], color='red', marker='o', s=8,
                    label='Target domain', alpha=0.8)

        # 添加图例和其他装饰
        plt.xticks(())
        plt.yticks(())

        out_dir = "/Workpalce_sdc/dxw/CDU-debug/tsne/CDU/Office31"

        plt.title('CDU' + ' (' + str(self.domains).upper() + ')', fontdict={"family": "Times New Roman", "size": 64})

        plt.savefig(out_dir + '/CDU-' + str(self.domains).upper() + '.pdf')

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        # self.T_SNE_combined()
        # print("compute source-target centroid distance:")
        # self.compute_distance()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            output = self.model(image)

            self.evaluator.process(output, label)

        if self.cfg.DATASET.NAME == "VisDA17":
            results, accs = self.evaluator.evaluate()
        else:
            results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
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
        #     return results["perclass_accuracy"]
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

        return list(results.values())[0]


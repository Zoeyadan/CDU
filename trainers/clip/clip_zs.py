import sys
from itertools import chain

import numpy as np
from clip.model import convert_weights
from matplotlib import pyplot as plt
from openTSNE import TSNE
from torch.cuda.amp import GradScaler

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip

from trainers.baseda import *
from utils.clip_part import *
from utils.templates import CUSTOM_TEMPLATES


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

def load_clip_to_cpu_teacher():
    backbone_name = "ViT-L/14"

    if backbone_name == "ViT-L/14":
        model_path = "assets/ViT-L-14.pt"
    elif backbone_name == "ViT-B/16":
        model_path = "assets/ViT-B-16.pt"
    else:
        print("teaher model name is false")
        sys.exit()

    print(f"CLIP Source name is {backbone_name}")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)

        clip_model_teacher = load_clip_to_cpu_teacher()
        self.text_encoder = Simple_TextEncoder(clip_model_teacher)

        self.image_encoder = clip_model.visual
        self.VPT_image_trans = Feature_Trans_Module_two_layer(clip_model.text_projection.shape[1], 768)
        self.device = torch.device("cuda:{}".format(cfg.GPU))
        self.VPT_image_trans = self.VPT_image_trans.to(self.device)
        convert_weights(self.VPT_image_trans)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts = clip.tokenize(prompts)
    
    def forward(self, image):
        text_features = self.text_encoder(self.tokenized_prompts.to(self.logit_scale.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = self.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t().cuda(image_features.device)

        return logits


@TRAINER_REGISTRY.register()
class CLIP_ZS(BaseDA):
    """
    ZS: Zero-Shot CLIP
    """  
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        output_dir = cfg.OUTPUT_DIR
        path_parts = output_dir.split('/')
        self.results_file ='/'.join(path_parts[:7])+ '/' + cfg.DATASET.NAME + ".csv"
        self.t_sne_path = '/'.join(path_parts[:6])

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CLIP.PREC == "fp32" or cfg.TRAINER.CLIP.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print("# params: {:,}".format(0))

        self.model.to(self.device)

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
            raise ValueError

        # no loss
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP_model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.CLIP.PREC == "amp" else None
    
    def train(self):
        self.before_train()
        self.after_train()

    @torch.no_grad()
    def T_SNE_combined(self):
        self.set_model_mode("eval")
        #
        all_embeddings = []
        all_labels = []

        combined_loader = chain(self.train_loader_x, self.train_loader_u)

        for batch_idx, batch in enumerate(combined_loader):
            input, label = self.parse_batch_test(batch)

            image_features = self.model.image_encoder(input.type(self.model.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(image_features.cpu().numpy())
            if batch_idx < len(self.train_loader_x):
                all_labels.extend([0] * len(label))
            else:
                all_labels.extend([1] * len(label))

        all_embeddings = np.vstack(all_embeddings)

        tsne = TSNE(perplexity=50, metric="euclidean", random_state=42)
        embeddings = tsne.fit(all_embeddings)

        source_mask = np.array(all_labels) == 0
        target_mask = np.array(all_labels) == 1

        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings[source_mask, 0], embeddings[source_mask, 1],
                    color='blue', marker='o', s=96, label='Source domain', alpha=0.5)
        plt.scatter(embeddings[target_mask, 0], embeddings[target_mask, 1],
                    color='red', marker='o', s=96, label='Target domain', alpha=0.5)

        plt.xticks(())
        plt.yticks(())

        print(self.t_sne_path + '/CLIP' + ' (' + self.domains.upper() + ')' + '.pdf')
        plt.title('CLIP' + ' (' + self.domains.upper() + ')', fontdict={"family": "Times New Roman", "size": 64})
        plt.savefig(self.t_sne_path + '/CLIP' + ' (' + self.domains.upper() + ')' + '.pdf')

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

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all
        
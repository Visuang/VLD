import torchvision
import torch.nn as nn
import torch
import copy
import numpy as np

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x



class PromptLearner(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()

        ctx_init = "A X X X X person observed in both day and night conditions."
        n_ctx = 1
        ctx_dim = 512
        ctx_init = ctx_init.replace("_", " ")

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self,):
        cls_ctx = self.cls_ctx
        prefix = self.token_prefix.expand(self.num_class, -1, -1)
        suffix = self.token_suffix.expand(self.num_class, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts





class Model(nn.Module):
    def __init__(self,config, num_classes, img_h, img_w):
        super(Model, self).__init__()
        self.config = config
        self.in_planes = 2048
        self.num_classes = num_classes

        self.h_resolution = int((img_h - 16) // 16 + 1)
        self.w_resolution = int((img_w - 16) // 16 + 1)
        self.vision_stride_size = 16
        clip_model,model_dict = load_clip_to_cpu('ViT-B-16', self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        self.classifier = Classifier(self.num_classes)
        self.classifier_proj = Classifier2(self.num_classes)

        self.classifier_STP = Classifier(self.num_classes)
        self.classifier_proj_STP = Classifier2(self.num_classes)

        self.text_encoder = TextEncoder(clip_model)

        self.prompt_learner = PromptLearner(num_classes, clip_model.dtype, clip_model.token_embedding)

        self.use_text_prompt_learning = True
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.seq_lenth = config.sequence_length

        self.prompt_embedding = nn.Parameter(torch.randn(self.seq_lenth, self.seq_lenth, 768))
        self.is_STA = config.is_STA
        self.STH_start_layer = config.STH_start_layer
        self.head_num = 12
        self.attn = nn.MultiheadAttention(768, self.head_num)
        self.ln_post2 = copy.deepcopy(self.image_encoder.ln_post)

    def encode_image(self, x: torch.Tensor, cv_emb=None):
        x = self.image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if cv_emb != None:
            x[:, 0] = x[:, 0] + cv_emb
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(11):
            if i==self.STH_start_layer:
                x = x.permute(1, 0, 2)
                x = self.STH(x,is_start=True)
                x = x.permute(1, 0, 2)
            elif i>self.STH_start_layer:
                x = self.STH(x)
            x = self.image_encoder.transformer.resblocks[i](x)

        x11 = x
        if self.STH_start_layer>0:
            x11 = self.STH(x11)

        x12 = self.image_encoder.transformer.resblocks[11](x11)
        x11 = x11.permute(1, 0, 2)  # LND -> NLD
        x12 = x12.permute(1, 0, 2)  # LND -> NLD

        if self.is_STA:
            x12_stp = self.STA(x12)

        x12 = self.image_encoder.ln_post(x12)
        if self.image_encoder.proj is not None:
            xproj = x12 @ self.image_encoder.proj
            xproj_stp = x12_stp @ self.image_encoder.proj

        return x11, x12, xproj,x12_stp,xproj_stp

    def STA(self,x12):
        pooled_output = x12[:, 0, :]
        bs = pooled_output.size(0) // self.seq_lenth
        D = pooled_output.size(-1)
        q = pooled_output.reshape(bs, self.seq_lenth, D).transpose(0, 1)
        kv = x12[:, -self.seq_lenth:, :].reshape(bs, self.seq_lenth * self.seq_lenth, D).transpose(0, 1)
        attn_output = self.attn(q, kv, kv)[0]
        pooled_output = attn_output.transpose(0, 1).reshape(bs * self.seq_lenth, D)
        pooled_output = self.image_encoder.ln_post(pooled_output)
        pooled_output = pooled_output.unsqueeze(1)
        return pooled_output

    def STH(self, input, is_start=False):
        if is_start:
            bs = input.size(0) // self.seq_lenth
            prompts = self.prompt_embedding.repeat(bs, 1, 1, 1).flatten(0, 1)  # [bs * n_prompt, prompt_len, dim]
            x = torch.cat([input, prompts], dim=1)
            return x
        else:
            bs = input.size(1) // self.seq_lenth
            Ls = input.size(0) - self.seq_lenth
            input = input.transpose(0, 1)  # [bs * seq_len, L, dim]

            hidden_states = input.reshape(bs, self.seq_lenth, Ls + self.seq_lenth, -1)  # [bs, seq_len, total_len, dim]
            prompts = hidden_states[:, :, Ls:, :].clone()  # last prompt_len positions
            hidden_states[:, :, Ls:, :] = prompts.transpose(1, 2)  # swap seq_len <-> prompt_len

            hidden_states = hidden_states.reshape(-1, Ls + self.seq_lenth, hidden_states.size(-1))  # [bs * seq_len, L, dim]
            return hidden_states.permute(1, 0, 2)  # [L, bs * seq_len, dim]



    def forward(self, x1=None, x2=None):
        if x1 is not None and x2 is not None:
            image_features_maps = torch.cat([x1, x2], dim=0)
            image_features_last, image_features, image_features_proj,image_features_stp, image_features_proj_stp = self.encode_image(image_features_maps)

            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]
            image_features_stp, image_features_proj_stp = image_features_stp[:,0], image_features_proj_stp[:,0]

            img_feature_last = self.tem_pool(img_feature_last)
            img_feature = self.tem_pool(img_feature)
            img_feature_proj = self.tem_pool(img_feature_proj)

            image_features_stp = self.tem_pool(image_features_stp)
            image_features_proj_stp = self.tem_pool(image_features_proj_stp)

            cls_scores, _ = self.classifier(img_feature)
            cls_scores_proj, _ = self.classifier_proj(img_feature_proj)

            cls_scores_stp, _ = self.classifier_STP(image_features_stp)
            cls_scores_proj_stp, _ = self.classifier_proj_STP(image_features_proj_stp)

            if self.use_text_prompt_learning:
                prompts = self.prompt_learner()
                tokenized_prompts = self.prompt_learner.tokenized_prompts
                text_features = self.text_encoder(prompts, tokenized_prompts)

                logits = img_feature_proj @ text_features.t()
                logits_stp = image_features_proj_stp@text_features.t()
            return [img_feature_last,img_feature,img_feature_proj,image_features_stp], [cls_scores, cls_scores_proj,logits,cls_scores_stp,logits_stp]

        else:
            image_features_last1, image_features1, image_features_proj1,image_features_stp, image_features_proj_stp = self.encode_image(x1)
            image_features1 = image_features1[:,0]
            image_features_stp = image_features_stp[:,0]
            image_features1 = self.tem_pool(image_features1)
            image_features_stp = self.tem_pool(image_features_stp)


            return torch.cat([image_features1,image_features_stp],dim=-1)

    def tem_pool(self,features,t=6):
        features = features.squeeze()
        features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
        features = features.mean(0)
        return features


class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        # self.GEM = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(768)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(768, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return cls_score, self.l2_norm(bn_features)

class Classifier2(nn.Module):
    def __init__(self, pid_num):
        super(Classifier2, self, ).__init__()
        self.pid_num = pid_num
        self.BN = nn.BatchNorm1d(512)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(512, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.l2_norm = Normalize(2)

    def forward(self, features):
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        return cls_score, self.l2_norm(features)


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model,model.state_dict()
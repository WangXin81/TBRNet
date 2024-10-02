import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm

import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.ASPP import ASPP
from model.PPM import PPM
from model.PSPNet import OneModel as PSPNet
from util.util import get_train_val_set


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def cos_similarity(main, aux):
    b, c, h, w = main.shape
    cosine_eps = 1e-7
    main = main.view(b, c, -1).permute(0, 2, 1).contiguous()  # [b, h*w, c]
    main_norm = torch.norm(main, 2, 2, True)
    aux = aux.view(b, c, -1)
    aux_norm = torch.norm(aux, 2, 1, True)

    logits = torch.bmm(main, aux) / (torch.bmm(main_norm, aux_norm) + cosine_eps)  # [b, hw, hw]
    similarity = torch.mean(logits, dim=-1).view(b, h * w)
    similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
            similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
    corr_query = similarity.view(b, 1, h, w)
    return corr_query

def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # b,C,N
    fea_T = fea.permute(0, 2, 1)  # b,N,C
    fea_norm = fea.norm(2, 2,
                        True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.print_freq = args.print_freq / 2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        elif self.dataset == 'iSAID':
            self.base_classes = 10
        elif self.dataset == 'DLRSD':
            self.base_classes = 10
        elif self.dataset == 'LoveDA':
            self.base_classes = 4

        assert self.layers in [50, 101, 152]


        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
        weight_path = 'initmodel/PSPNet/split{}/{}/best.pth'.format(args.split, backbone_str)
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:  # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 4, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        # FEM
        self.pyramid_bins = [60, 30, 15, 8]
        self.avgpool_list = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.avgpool_list.append(
                nn.AdaptiveAvgPool2d(bin)
            )
        self.init_merge_FEM = []
        for bin in self.pyramid_bins:
            self.init_merge_FEM.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
            ))
        self.init_merge_FEM = nn.ModuleList(self.init_merge_FEM)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))

        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

        self.con1 = nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.con2 = nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.con3 = nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.con4 = nn.Sequential(
            nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim,
                    kernel_size=1, stride=1, padding=0),
            BatchNorm(reduce_dim)
        )
        nn.init.constant_(self.con4[1].weight, 0)
        nn.init.constant_(self.con4[1].bias, 0)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def get_optim(self, model, args, LR):
        if args.shot > 1:
            optimizer = torch.optim.SGD(
                [
                    {'params': model.down_query.parameters()},
                    {'params': model.down_supp.parameters()},
                    {'params': model.init_merge.parameters()},
                    {'params': model.ASPP_meta.parameters()},
                    {'params': model.res1_meta.parameters()},
                    {'params': model.res2_meta.parameters()},
                    {'params': model.cls_meta.parameters()},
                    {'params': model.gram_merge.parameters()},
                    {'params': model.cls_merge.parameters()},
                    {'params': model.kshot_rw.parameters()},
                    {'params': model.beta_conv.parameters()},
                    {'params': model.inner_cls.parameters()},
                    {'params': model.init_merge_FEM.parameters()},
                    {'params': model.alpha_conv.parameters()},
                    {'params': model.con2.parameters()},
                    {'params': model.con3.parameters()},
                    {'params': model.con1.parameters()},
                    {'params': model.con4.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [
                    {'params': model.down_query.parameters()},
                    {'params': model.down_supp.parameters()},
                    {'params': model.init_merge.parameters()},
                    {'params': model.ASPP_meta.parameters()},
                    {'params': model.res1_meta.parameters()},
                    {'params': model.res2_meta.parameters()},
                    {'params': model.cls_meta.parameters()},
                    {'params': model.gram_merge.parameters()},
                    {'params': model.cls_merge.parameters()},
                    {'params': model.beta_conv.parameters()},
                    {'params': model.inner_cls.parameters()},
                    {'params': model.init_merge_FEM.parameters()},
                    {'params': model.alpha_conv.parameters()},
                    {'params': model.con2.parameters()},
                    {'params': model.con3.parameters()},
                    {'params': model.con1.parameters()},
                    {'params': model.con4.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False

    def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None):
        x_size = x.size()
        bs = x_size[0]  # batch_size
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)  # f(q,m)

        # Support Feature
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)  # shot x [bs, 1, h , w]
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)  # f(s,m)

            supp_feat = self.PRM(supp_feat, query_feat)
            supp_pro = Weighted_GAP(supp_feat, mask)
            supp_pro_list.append(supp_pro)
            supp_feat_list.append(eval('supp_feat_' + self.low_fea_id))

        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id))  # [bs, C, C]

        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        base_out = self.learner_base(query_feat_4)  # bs,11,30,30

        res = []
        cosine_eps = 1e-7
        bins = [60, 30, 15, 8]
        main = query_feat_4
        tmp = []
        if main.shape[-1] == 30:  # vgg
            for k, tmp_supp_feat in enumerate(final_supp_list):
                temp_aux = tmp_supp_feat
                temp_main = F.interpolate(main, size=(60, 60), mode='bilinear', align_corners=True)
                temp_aux = F.interpolate(temp_aux, size=temp_main.shape[-2:], mode='bilinear', align_corners=True)
                temp_mask = F.interpolate(mask, size=temp_main.shape[-2:], mode='bilinear', align_corners=True)
                temp_aux = temp_aux * temp_mask
                sim_i = cos_similarity(temp_main, temp_aux)
                tmp.append(sim_i)
            corr_query_mask_vgg = torch.cat(tmp, 1)
            corr_query_mask_vgg = (weight_soft * corr_query_mask_vgg).sum(1, True)  # bs,1,30,30
            res.append(corr_query_mask_vgg)
            bins = bins[1:]
        for j in range(len(bins)):
            corr_query_mask_list = []
            for i, tmp_supp_feat in enumerate(final_supp_list):
                resize_size = main.size(2)
                tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear',
                                         align_corners=True)
                tmp_supp_feat = F.interpolate(tmp_supp_feat, size=(resize_size, resize_size), mode='bilinear',
                                              align_corners=True)
                tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                q = main
                s = tmp_supp_feat_4
                bsize, ch_sz, sp_sz, _ = q.size()[:]

                tmp_query = q
                tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
                tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                tmp_supp = s
                tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
                tmp_supp = tmp_supp.permute(0, 2, 1)
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)  # sim_i
                corr_query_mask_list.append(corr_query)
            corr_query_mask = torch.cat(corr_query_mask_list, 1)
            corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)  # bs,1,30,30

            res.append(corr_query_mask)
            main = main * corr_query_mask
            if j != len(bins) - 1:
                main = F.adaptive_avg_pool2d(main, bins[j + 1])

        # Support Prototype
        supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
        supp_pro = (weight_soft.permute(0, 2, 1, 3) * supp_pro).sum(2, True)  # bs,C,1,1

        out_list = []
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            bin = tmp_bin
            query_feat_bin = self.avgpool_list[idx](query_feat)
            supp_feat_bin = supp_pro.expand(-1, -1, bin, bin)
            corr_mask_bin = res[idx]
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge_FEM[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        query_meta = torch.cat(pyramid_feat_list, 1)
        query_meta = self.init_merge(query_meta)
        query_meta = self.ASPP_meta(query_meta)
        query_meta = self.res1_meta(query_meta)
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        meta_map_bg = meta_out_soft[:, 0:1, :, :]
        meta_map_fg = meta_out_soft[:, 1:, :, :]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes + 1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
            base_map = torch.cat(base_map_list, 0)
        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))


        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)  # bs,2,473,473
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)  # bs,11,473,473
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)  # bs,2,473,473

        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_loss2 = self.criterion(base_out, y_b.long())

            aux_loss = torch.zeros_like(main_loss).cuda()
            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y_m.long())
            aux_loss = aux_loss / len(out_list)
            return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2, aux_loss
        else:
            return final_out, meta_out, base_out

    def PRM(self, s, q):
        batch_size = s.size(0)
        inter_channels = 256

        v_x = self.con1(q).view(batch_size, inter_channels, -1)
        v_x = v_x.permute(0, 2, 1)  # bs,hw,C

        q_x = self.con2(s).view(batch_size, inter_channels, -1)
        q_x = q_x.permute(0, 2, 1) # bs,hw,C

        k_x = self.con3(q).view(batch_size, inter_channels, -1) # bs,C,hw

        f = torch.matmul(q_x, k_x) # bs,hw,hw
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, v_x) # bs,hw,C
        y = y.permute(0, 2, 1).contiguous() # bs,C,hw
        y = y.view(batch_size, inter_channels, *s.size()[2:]) # bs,C,h,w

        W_y = self.con4(y)

        q = (self.gap(q).view(q.size(0), -1, 1, 1)).expand_as(s)
        mask = torch.cosine_similarity(s, q, dim=1).unsqueeze(1)
        z = W_y * mask.expand_as(W_y) + s  # bs,C,h,w

        return z
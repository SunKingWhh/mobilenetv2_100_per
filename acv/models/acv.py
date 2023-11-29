from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
import gc
import time


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        gwc_feature = torch.cat((l2, l3, l4), dim=1) #16个残差后连接了l1\l2\l3 320维  MAPM模块
        return {"gwc_feature": gwc_feature}

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6

class ACVNet(nn.Module):
    def __init__(self, maxdisp, attn_weights_only, freeze_attn_weights):
        super(ACVNet, self).__init__()
        self.maxdisp = maxdisp
        self.attn_weights_only = attn_weights_only
        self.freeze_attn_weights = freeze_attn_weights
        self.num_groups = 40
        self.concat_channels = 32
        self.feature_extraction = feature_extraction() #返回320维
        self.concatconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

        self.patch = nn.Conv3d(40, 40, kernel_size=(1,3,3), stride=1, dilation=1, groups=40, padding=(0,1,1), bias=False)
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1,3,3), stride=1, dilation=1, groups=8, padding=(0,1,1), bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=2, groups=16, padding=(0,2,2), bias=False)
        self.patch_l3 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=3, groups=16, padding=(0,3,3), bias=False)

        self.dres1_att_ = nn.Sequential(convbn_3d(40, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
        self.dres2_att_ = hourglass(32)
        self.classif_att_ = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))        
        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):

        if self.freeze_attn_weights: # 首次训练不运行    第二部，冻结自注意力网络，训练其他的    第三步训练不运行
            with torch.no_grad(): #不计算梯度
                features_left = self.feature_extraction(left)#提特征 返回320维
                features_right = self.feature_extraction(right)
                gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)#构建体积 320维->40组=40维
                gwc_volume = self.patch(gwc_volume) # 分成40组
                patch_l1 = self.patch_l1(gwc_volume[:, :8])# l1分组
                patch_l2 = self.patch_l2(gwc_volume[:, 8:24]) # l2分组
                patch_l3 = self.patch_l3(gwc_volume[:, 24:40]) # l3分组
                patch_volume = torch.cat((patch_l1,patch_l2,patch_l3), dim=1) #  多尺度自注意力代价体积
                cost_attention = self.dres1_att_(patch_volume)#两次3D卷积
                cost_attention = self.dres2_att_(cost_attention)# 沙漏网络卷积
                att_weights = self.classif_att_(cost_attention)#变成1维的自注意力权重

        else:
            # 首次训练，自注意力网络，生成自注意力权重    第三步训练运行
            features_left = self.feature_extraction(left) #提特征
            features_right = self.feature_extraction(right)
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups) #构建体积 320维->40组=40维
            gwc_volume = self.patch(gwc_volume) #按组卷积
            patch_l1 = self.patch_l1(gwc_volume[:, :8]) # l1分组
            patch_l2 = self.patch_l2(gwc_volume[:, 8:24]) # l2分组
            patch_l3 = self.patch_l3(gwc_volume[:, 24:40]) # l3分组
            patch_volume = torch.cat((patch_l1,patch_l2,patch_l3), dim=1)
            cost_attention = self.dres1_att_(patch_volume) #两次3D卷积
            cost_attention = self.dres2_att_(cost_attention)# 沙漏网络卷积
            att_weights = self.classif_att_(cost_attention) #变成1维的自注意力权重

        if not self.attn_weights_only:# 首轮训练不运行  第二步运行 第三步训练运行
            concat_feature_left = self.concatconv(features_left["gwc_feature"]) #构建体积 32维   通过concatconv，两次卷积卷成32维
            concat_feature_right = self.concatconv(features_right["gwc_feature"])  #构建体积 32维
            concat_volume = build_concat_volume(concat_feature_left, concat_feature_right, self.maxdisp // 4)#构建初始体积 32+32=64
            print("concat_volume",concat_volume.size())
            ac_volume = F.softmax(att_weights, dim=2) * concat_volume   # ac_volume = att_weights * concat_volume
            cost0 = self.dres0(ac_volume)
            cost0 = self.dres1(cost0) + cost0
            out1 = self.dres2(cost0)
            out2 = self.dres3(out1)

        if self.training:

            if not self.freeze_attn_weights: # 首次训练运行 第二步不运行 第三步训练运行

                cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')#上采样到指定的尺寸
                cost_attention = torch.squeeze(cost_attention, 1)# 删除张量中的单个维度（如果该维度的大小为1）
                pred_attention = F.softmax(cost_attention, dim=1)# 将输入向量映射为输出向量的概率分布，使得输出向量的每个元素都介于0到1之间，且所有元素之和为1。
                pred_attention = disparity_regression(pred_attention, self.maxdisp)

            if not self.attn_weights_only:# 第二步训练输出 第三步训练运行

                cost0 = self.classif0(cost0)
                cost1 = self.classif1(out1)
                cost2 = self.classif2(out2)    

                cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost0 = torch.squeeze(cost0, 1)
                pred0 = F.softmax(cost0, dim=1)
                pred0 = disparity_regression(pred0, self.maxdisp)    

                cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost1 = torch.squeeze(cost1, 1)
                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparity_regression(pred1, self.maxdisp)    

                cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost2 = torch.squeeze(cost2, 1)
                pred2 = F.softmax(cost2, dim=1)
                pred2 = disparity_regression(pred2, self.maxdisp)

                if self.freeze_attn_weights: # 第二步训练运行   第三步训练不运行
                    return [pred0, pred1, pred2] # 第二步 返回值
                return [pred_attention, pred0, pred1, pred2]# 第三步 返回值
            return [pred_attention] # 首次训练返回值，自注意力权重

        else: # 测试时输出

            if self.attn_weights_only:

                cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost_attention = torch.squeeze(cost_attention, 1)
                pred_attention = F.softmax(cost_attention, dim=1)
                pred_attention = disparity_regression(pred_attention, self.maxdisp)
                return [pred_attention]

            cost2 = self.classif2(out2)
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            return [pred2]

def acv(d):
    return ACVNet(d)

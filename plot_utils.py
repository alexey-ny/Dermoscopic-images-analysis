# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:08:43 2023

@author: alex
"""

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# channels = ['3', 'EI', 'MI', '5']

def plot_mean_ROC(target, predictions, channels, model_descr = '', to_dir = '', channels_first = True):
    color_index = list(mcolors.CSS4_COLORS)
    colors = mcolors.CSS4_COLORS
    # color_index = list(mcolors.TABLEAU_COLORS)
    # colors = mcolors.TABLEAU_COLORS
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for n_ch, ch in enumerate(channels):  
        fig, ax = plt.subplots(figsize=(6, 6), dpi = 200)
        for fold in range(5):
            # print(fold, n_ch, ch, target.shape, predictions[ch][fold].shape, predictions[ch].T[fold].shape, predictions[ch].shape)
            if channels_first:
                fpr, tpr, t = roc_curve(target, predictions[ch][fold])
                # fpr, tpr, t = roc_curve(y_test_set, oof_pred[ch][fold])
                f_auc = roc_auc_score(target, predictions[ch][fold])
            # f_auc = roc_auc_score(y_test_set, oof_pred[ch][fold])
            else:
                fpr, tpr, t = roc_curve(target, predictions[ch].T[fold])
                f_auc = roc_auc_score(target, predictions[ch].T[fold])
                
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            # aucs.append(viz.roc_auc)
            aucs.append(f_auc )
            ax.plot(
                fpr,
                tpr,
                color = colors[color_index[fold]],
                label = f"ROC fold {fold + 1} AUC = {f_auc:.4f}",
                lw=2,
                alpha = 0.3,
            )
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color = colors[color_index[fold + 1]],
            label=r"Mean ROC (AUC = %0.4f $\pm$ %0.4f)" % (mean_auc, std_auc),
            lw=2,
            alpha = 0.95,
        )
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability\n for {ch} channels",
            # title=f"Mean ROC curve with variability\n(Positive label '{target_names[1]}')",
        )
        ax.axis("square")
        ax.legend(loc="lower right")
        if len(to_dir)>0:
            if not os.path.exists(f'{to_dir}'):
                os.mkdir(f'{to_dir}')
            plt.savefig(f'{to_dir}/{model_descr}_ROC_{ch}_channels.png')
            # if not os.path.exists(f'{to_dir}/{model_descr}'):
            #     os.mkdir(f'{to_dir}/{model_descr}')
            # plt.savefig(f'{to_dir}/{model_descr}/{model_descr}_AUC_{ch}_channels.png')
        plt.show()


def plot_mean_ROC_multi_models(target, predictions, channels, model_descr = '', to_dir = ''):
    color_index = list(mcolors.CSS4_COLORS)
    colors = mcolors.CSS4_COLORS
    # color_index = list(mcolors.TABLEAU_COLORS)
    # colors = mcolors.TABLEAU_COLORS
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for n_ch, ch in enumerate(channels):  
        fig, ax = plt.subplots(figsize=(6, 6), dpi = 200)
        for fold in range(len(predictions[ch])):
        # for fold in range(5):
            fpr, tpr, t = roc_curve(target, predictions[ch][fold])
            # fpr, tpr, t = roc_curve(y_test_set, oof_pred[ch][fold])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            f_auc = roc_auc_score(target, predictions[ch][fold])
            # f_auc = roc_auc_score(y_test_set, oof_pred[ch][fold])
            tprs.append(interp_tpr)
            # aucs.append(viz.roc_auc)
            aucs.append(f_auc )
            ax.plot(
                fpr,
                tpr,
                color = colors[color_index[fold]],
                # label = f"ROC fold {fold + 1} AUC = {f_auc:.4f}",
                lw=2,
                alpha = 0.3,
            )
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color = colors[color_index[fold + 1]],
            label=r"Mean ROC (AUC = %0.4f $\pm$ %0.4f)" % (mean_auc, std_auc),
            lw=2,
            alpha = 0.95,
        )
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability\n for {ch} channels",
            # title=f"Mean ROC curve with variability\n(Positive label '{target_names[1]}')",
        )
        ax.axis("square")
        ax.legend(loc="lower right")
        if len(to_dir)>0:
            if not os.path.exists(f'{to_dir}'):
                os.mkdir(f'{to_dir}')
            plt.savefig(f'{to_dir}/{model_descr}_ROC_{ch}_channels.png')
            # if not os.path.exists(f'{to_dir}/{model_descr}'):
            #     os.mkdir(f'{to_dir}/{model_descr}')
            # plt.savefig(f'{to_dir}/{model_descr}/{model_descr}_AUC_{ch}_channels.png')
        plt.show()


def plot_PR(target, predictions, channels):
    color_index = list(mcolors.CSS4_COLORS)
    colors = mcolors.CSS4_COLORS
    # color_index = list(mcolors.TABLEAU_COLORS)
    # colors = mcolors.TABLEAU_COLORS
    
    aucs = []  
    for n_ch, ch in enumerate(channels):  
        fig, ax = plt.subplots(figsize=(6, 6))
        for fold in range(5):
            f_pre, f_rec, t = precision_recall_curve(target, predictions[ch][fold])
            f_f1 = metrics.f1_score(target, predictions[ch][fold] >= 0.5)
            pr_auc = metrics.auc(f_rec, f_pre)
            aucs.append(pr_auc)
            ax.plot(
                f_rec,
                f_pre,
                color = colors[color_index[fold]],
                label = f"PR fold {fold + 1}, AUC = {pr_auc:.4f}, F1 = {f_f1:0.4f}",
                lw=2,
                alpha = 0.3,
            )
                
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="Recall",
            ylabel="Precision",
            title=f"PR curves for 5 folds\n for {ch} channels",
        )
        ax.axis("square")
        ax.legend(loc="lower right")
        plt.show()


def plot_mean_PR(target, predictions, channels, model_descr = '', to_dir = '', channels_first = True):
    color_index = list(mcolors.CSS4_COLORS)
    colors = mcolors.CSS4_COLORS
    # color_index = list(mcolors.TABLEAU_COLORS)
    # colors = mcolors.TABLEAU_COLORS
    
    precisions = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)
    for n_ch, ch in enumerate(channels):  
        # fig, ax = plt.subplots(figsize=(6, 6))
        fig, ax = plt.subplots(figsize=(6, 6), dpi = 200)
        for fold in range(5):
            if channels_first:
                f_pre, f_rec, t = precision_recall_curve(target, predictions[ch][fold])
                f_f1 = metrics.f1_score(target, predictions[ch][fold] >= 0.5)
                # f_pre, f_rec, t = precision_recall_curve(y_test_set, oof_pred[ch][fold])
                # f_f1 = metrics.f1_score(y_test_set, oof_pred[ch][fold] >= 0.5)
            else:
                f_pre, f_rec, t = precision_recall_curve(target, predictions[ch].T[fold])
                f_f1 = metrics.f1_score(target, predictions[ch].T[fold] >= 0.5)
            interp_precision = np.interp(mean_recall, np.flip(f_rec), np.flip(f_pre))
            # interp_precision = np.interp(mean_recall, f_rec, f_pre)
            # interp_tpr = np.interp(mean_fpr, f_rec, f_pre)
            # interp_tpr[0] = 0.0
            pr_auc = metrics.auc(f_rec, f_pre)
            # pr_auc = metrics_auc(target, predictions[ch][fold])
            # f_auc = roc_auc_score(y_test_set, oof_pred[ch][fold])
            precisions.append(interp_precision)
            aucs.append(pr_auc)
            ax.plot(
                f_rec,
                f_pre,
                color = colors[color_index[fold]],
                label = f"PR fold {fold + 1}, AUC = {pr_auc:.4f}, F1 = {f_f1:0.4f}",
                lw=2,
                alpha = 0.3,
            )
        
        mean_precision = np.mean(precisions, axis=0)
        # mean_pre = np.mean(f_recs, axis=0)
        # mean_rec = np.mean(f_pres, axis=0)
        # mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_recall, mean_precision)
        # mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_recall,
            mean_precision,
            # mean_fpr,
            # mean_tpr,
            color = colors[color_index[fold + 1]],
            label=r"Mean PR (AUC = %0.4f $\pm$ %0.4f)" % (mean_auc, std_auc),
            lw=2,
            alpha = 0.95,
        )
        
        std_precision = np.std(precisions, axis=0)
        # std_tpr = np.std(tprs, axis=0)
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_recall,
            # mean_fpr,
            precisions_lower,
            precisions_upper,
            # tprs_lower,
            # tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="Recall",
            ylabel="Precision",
            title=f"PR curves for 5 folds\n for {ch} channels",
        )
        ax.axis("square")
        ax.legend(loc="upper right")
        if len(to_dir)>0:
            if not os.path.exists(f'{to_dir}'):
                os.mkdir(f'{to_dir}')
            plt.savefig(f'{to_dir}/{model_descr}_PRC_{ch}_channels.png')
            # if not os.path.exists(f'{to_dir}/{model_descr}'):
            #     os.mkdir(f'{to_dir}/{model_descr}')
            # plt.savefig(f'{to_dir}/{model_descr}/{model_descr}_PRC_{ch}_channels.png')
        plt.show()


def plot_mean_PR_multi_models(target, predictions, channels, model_descr = '', to_dir = ''):
    color_index = list(mcolors.CSS4_COLORS)
    colors = mcolors.CSS4_COLORS
    # color_index = list(mcolors.TABLEAU_COLORS)
    # colors = mcolors.TABLEAU_COLORS
    
    precisions = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)
    for n_ch, ch in enumerate(channels):  
        fig, ax = plt.subplots(figsize=(6, 6), dpi = 200)
        for fold in range(len(predictions[ch])):
        # for fold in range(5):
            f_pre, f_rec, t = precision_recall_curve(target, predictions[ch][fold])
            f_f1 = metrics.f1_score(target, predictions[ch][fold] >= 0.5)
            interp_precision = np.interp(mean_recall, np.flip(f_rec), np.flip(f_pre))
            pr_auc = metrics.auc(f_rec, f_pre)
            precisions.append(interp_precision)
            aucs.append(pr_auc)
            ax.plot(
                f_rec,
                f_pre,
                color = colors[color_index[fold]],
                # label = f"PR fold {fold + 1}, AUC = {pr_auc:.4f}, F1 = {f_f1:0.4f}",
                lw=2,
                alpha = 0.3,
            )
        
        mean_precision = np.mean(precisions, axis=0)
        mean_auc = metrics.auc(mean_recall, mean_precision)
        std_auc = np.std(aucs)
        ax.plot(
            mean_recall,
            mean_precision,
            color = colors[color_index[fold + 1]],
            label=r"Mean PR (AUC = %0.4f $\pm$ %0.4f)" % (mean_auc, std_auc),
            lw=2,
            alpha = 0.95,
        )
        
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(
            mean_recall,
            precisions_lower,
            precisions_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="Recall",
            ylabel="Precision",
            title=f"PR curves for 5 folds\n for {ch} channels",
        )
        ax.axis("square")
        ax.legend(loc="upper right")
        if len(to_dir)>0:
            if not os.path.exists(f'{to_dir}'):
                os.mkdir(f'{to_dir}')
            plt.savefig(f'{to_dir}/{model_descr}_PRC_{ch}_channels.png')
        plt.show()

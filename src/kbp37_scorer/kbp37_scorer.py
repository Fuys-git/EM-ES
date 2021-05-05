import sys
import os
import json
import re
from collections import defaultdict
import argparse

def change_label(label):
    if label == "no_relation":
        return "Other"
    label = label.replace(":", "").replace("/","")
    if "_" in label:
        label = label.replace("_", "-")
    else:
        segs = label.split("(", 1)
        label = "%s-%s(%s" % (segs[0], segs[0], segs[1])
    return label

def get_gold_id2label(gold_file, out_fp = None):

    sen_id_li = []
    id2label_dict = dict()

    with open(gold_file, 'r') as f:
        datas = f.readlines()
    for data in datas:
        data = data[:-1].split('\t')
        sen_id = data[0]
        label = data[1]
        sen_id_li.append(sen_id)
        id2label_dict[sen_id] = label
    if out_fp is not None:
        fout = open(out_fp, "w")
        for sen_id in sen_id_li:
            label = id2label_dict[sen_id]
            fout.write("%s\t%s\n" % (sen_id, label))
        fout.close()
    return sen_id_li, id2label_dict


def get_pred_id2label(pred_fp, sen_id_li, id2label_dict, out_fp=None):
    i = 0
    pred_id2label_dict = dict()
    print(pred_fp)
    for line in open(pred_fp):
        segs = line[:-1].split("\t")
        if len(segs) != 2:
            continue
        else:
            sen_id = sen_id_li[i]
            _pre_lab_id = int(float(segs[1]))
            #print(".._pred_lab_id:%d" % _pre_lab_id)
            #print("22 in id2labe_dict:%s" % (22 in id2label_dict))
            if _pre_lab_id not in id2label_dict:
                sys.stderr.write(">>illegal _pre_lab_id %d rewrite to 1\n" % (_pre_lab_id))
                _pre_lab_id = 1
            label = id2label_dict[_pre_lab_id]
            pred_id2label_dict[sen_id] = label
            i += 1

    assert len(sen_id_li) == len(pred_id2label_dict), ">>expect len(sen_id_li) == len(pred_id2label_dict), but {} != {}".format(len(sen_id_li), len(pred_id2label_dict))
    if out_fp is not None:
        fout = open(out_fp, "w")
        for sen_id in sen_id_li:
            label = pred_id2label_dict[sen_id]
            fout.write("%s\t%s\n" % (sen_id, label))
        fout.close()
    return pred_id2label_dict

def load_id2label_dict(label_dict_fp, ignore_label):
    label_dict = json.load(open(label_dict_fp))
    id2label_dict = dict()
    for k, v in label_dict.items():
        id2label_dict[int(v)] = k
    srt_id2label_list = sorted(id2label_dict.items(), key = lambda x: x[0])

    display_label_list = []
    display_label_list_nodir = []
    display_label_list.append(ignore_label)
    display_label_list_nodir.append(ignore_label)

    for x in srt_id2label_list:
        if x[1] != ignore_label:
            display_label_list.append(x[1])
            lable_no_dir = re.sub("\(e[21],e[12]\)", "", x[1])
            if lable_no_dir not in display_label_list_nodir:
                display_label_list_nodir.append(lable_no_dir)
    return id2label_dict, display_label_list, display_label_list_nodir


def get_label_nodirction(label, ignore_label):
    if label == ignore_label:
        return label
    else:
        segs = label.split("(e", 1)
        assert len(segs) == 2
        return segs[0]

def _print_divide(numerator, denominator, use_percentage=True, sep="="):
    if denominator == 0:
        return "Zero division ERROR!"
    if use_percentage is True:
        outs = "%.2f%%%s%d/%d" % (100.0 * numerator / denominator, sep, numerator, denominator)
    else:
        outs = "%.2f%s%d/%d" % (numerator / denominator, sep, numerator, denominator)
    return outs

def kbp37_scorer(pred_test_fp, gold_test_fp, label_dict_fp, ignore_label, results_file, fout=sys.stdout):
    id2label_dict, label_list, label_list_nodir = load_id2label_dict(label_dict_fp, ignore_label)
    #print(">>label id_to_label len:%d" % len(id2label_dict))
    fout.write(">>label id_to_label len:%d\n" % len(id2label_dict))
    #sen_id_li, gold_id2label_dict = get_gold_id2label(gold_test_fp, out_fp="./tmp_gold_test.txt")
    sen_id_li, gold_id2label_dict = get_gold_id2label(gold_test_fp, out_fp=None)
    #pred_id2label_dict = get_pred_id2label(pred_test_fp, sen_id_li, id2label_dict, out_fp="./tmp_pred_test.txt")
    pred_id2label_dict = get_pred_id2label(pred_test_fp, sen_id_li, id2label_dict, out_fp=None)

    #fill in the confustion matrix
    conf_matrix_37way = defaultdict(lambda :defaultdict(int))   #37 = 18 *2 + 1

    pred_labelSum_37way = defaultdict(int)
    gold_labelSum_37way = defaultdict(int)

    conf_matrix_19way_withDir = defaultdict(lambda :defaultdict(int))  #19 = 18 + 1
    pred_labelSum_19way = defaultdict(int)
    gold_labelSum_19way_withDir = defaultdict(int)

    totalSum = len(pred_id2label_dict)

    jj = 0
    for sen_id in sen_id_li:
        jj += 1
        gold_lable = gold_id2label_dict[sen_id]
        pred_lable = pred_id2label_dict[sen_id]
        conf_matrix_37way[pred_lable][gold_lable] += 1
        pred_labelSum_37way[pred_lable] += 1
        gold_labelSum_37way[gold_lable] += 1

        gold_label_nodir = re.sub("\(e[21],e[12]\)", "", gold_lable)
        pred_label_nodir = re.sub("\(e[21],e[12]\)", "", pred_lable)

        #print("..pred_label:%s\tgold_label:%s" % (pred_lable, gold_lable))
        if (pred_lable != gold_lable) and (pred_label_nodir == gold_label_nodir):
            conf_matrix_19way_withDir["WRONG_DIR"][gold_label_nodir] += 1
            pred_labelSum_19way["WRONG_DIR"] += 1
        else:
            conf_matrix_19way_withDir[pred_label_nodir][gold_label_nodir] += 1
            pred_labelSum_19way[pred_label_nodir] += 1

        #Calculate the ground truth distributions
        gold_labelSum_37way[gold_lable] += 1
        gold_labelSum_19way_withDir[gold_label_nodir] += 1
        #if jj > 10:
        #    sys.exit(0)

    #compute the P/R/F1 for each label
    #print head line
    fout.write('>>18 way with direction (no "other")\n')
    fout.write("gold\pred|\t%s\t<-- classified as\n" % ("\t".join(label_list_nodir)))
    fout.write(" +\t%s\t+%s\n" % ("\t".join(["-"] * len(label_list_nodir)),
                          "\t".join(["SUM", "xDIRx", "skip", "ACTUAL"])))

    sum_otherSkiped = 0
    freq_correct = 0
    other_skipped = 0
    for i, gold_label in enumerate(label_list_nodir):   #row label
        out_segs = []
        out_segs.append(gold_label)
        sum_pred = 0
        for j, pred_label in enumerate(label_list_nodir):   #column label  conf[pred][gold]
            #[pred_gold][gold]
            cnt = conf_matrix_19way_withDir[pred_label].get(gold_label, 0)
            #cnt = conf_matrix_19way_withDir[gold_label].get(pred_label, 0)
            out_segs.append(str(cnt))
            sum_pred += cnt
        out_segs.append(str(sum_pred))

        wrongdir_cnt = conf_matrix_19way_withDir["WRONG_DIR"].get(gold_label, 0)
        out_segs.append(str(wrongdir_cnt))

        ans =gold_labelSum_19way_withDir.get(gold_label, 0)

        skip_cnt = ans - sum_pred - wrongdir_cnt
        out_segs.append(str(skip_cnt))

        out_segs.append(str(ans))
        fout.write("\t".join(out_segs) + "\n")
        if gold_label == ignore_label:
            other_skipped = skip_cnt

        sum_otherSkiped += skip_cnt
        freq_correct += conf_matrix_19way_withDir[gold_label].get(gold_label, 0)

    fout.write(" +\t%s\t+\n" % "\t".join(["-"] * len(label_list)))
    ### 3. Print the vertical sums
    out_segs = []
    out_segs.append("SUM")
    _pred_cnt_li = [str(pred_labelSum_19way.get(x, 0)) for x in label_list_nodir]
    out_segs.extend(_pred_cnt_li)

    _wrong_cnt = pred_labelSum_19way.get("WRONG_DIR", 0)
    out_segs.append(str(len(pred_id2label_dict) - _wrong_cnt))
    out_segs.append(str(_wrong_cnt))
    out_segs.append(str(len(gold_id2label_dict) - len(pred_id2label_dict)))
    out_segs.append(str(len(gold_id2label_dict)))
    fout.write("\t\n".join(out_segs))


    fout.write("\n\n")
    fout.write(">>Coverage\t%s\n" % (_print_divide(len(pred_id2label_dict), len(gold_id2label_dict), sep="\t")))
    fout.write(">>Accuracy\t%s\n" % _print_divide(freq_correct, len(pred_id2label_dict), sep="\t"))
    fout.write("\nResults for the individual relations:\nn")

    p_li = []
    r_li = []
    f1_li = []
    for i, gold_label in enumerate(label_list_nodir):
        ### 8.1. Consider all wrong directionalities as wrong classification decisions
        wrongDirectionCnt = conf_matrix_19way_withDir["WRONG_DIR"].get(gold_label, 0)

        ### 8.3. Calculate P/R/F1
        pre_labcnt = pred_labelSum_19way.get(gold_label, 0)
        P = 100.0 * conf_matrix_19way_withDir[gold_label][gold_label] / (pre_labcnt + wrongDirectionCnt) if pre_labcnt != 0 else 0.0

        gold_labcnt = gold_labelSum_19way_withDir.get(gold_label, 0)
        R = 100.0 * conf_matrix_19way_withDir[gold_label][gold_label] / (gold_labcnt) if gold_labcnt != 0 else 0.0

        F1 = 2 * P * R / (P + R) if (P + R > 0.0 ) else 0
        out_segs = [gold_label]
        p_str = "P\t%.2f%%\t=%4d/(%4d +%4d)" % (P, conf_matrix_19way_withDir[gold_label][gold_label], pre_labcnt, wrongDirectionCnt)
        r_str = "R\t%.2f%%\t=%4d/ %4d" % (R, conf_matrix_19way_withDir[gold_label][gold_label], gold_labcnt)
        f1_str = "F1\t%.2f" % F1
        out_segs.append(p_str)
        out_segs.append(r_str)
        out_segs.append(f1_str)
        #print("\t".join(out_segs))
        fout.write("\t".join(out_segs) + "\n")
        if gold_label != ignore_label:
            p_li.append(P)
            r_li.append(R)
            f1_li.append(F1)
    macor_p = sum(p_li) / len(p_li)
    macor_r = sum(r_li) / len(r_li)
    macor_f1 = sum(f1_li) / len(f1_li)
    fout.write(">>The official score macor_p\t%.4f\tmacro_r\t%.4f\tmacro_f1\t%.4f\n" %
          (macor_p, macor_r, macor_f1))

    # print results to txt file
    with open(results_file, 'a') as f:
        f.write(">>The official score of %s\tmacor_p\t%.4f\tmacro_r\t%.4f\tmacro_f1\t%.4f\n" %
                (gold_test_fp, macor_p, macor_r, macor_f1))

    return macor_p, macor_r, macor_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str,  default='./pred_labels_epoch7.txt')
    parser.add_argument("--golden_file", type=str, default='./true_labels_epoch7.txt')
    parser.add_argument("--label_dict", type=str, default='./idx2rel.json')
    parser.add_argument("--results_file", type=str, default='./results.txt')
    args = parser.parse_args()

    ignore_label = "no_relation"

    kbp37_scorer(args.pred_file, args.golden_file, args.label_dict, ignore_label, args.results_file)

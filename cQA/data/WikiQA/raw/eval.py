import os, sys


def get_prf(fref, fpred, thre=0.1):
    """
    get Q-A level and Q level precision, recall, fmeasure
    """
    ref, pred = [], []
    qref, qpred_idx = [], []
    preqpos = ""
    qflag = False
    with open(fref, "rb") as f:
        for line in f:
            parts = line.strip().split()
            qpos, l = parts[0], int(parts[3])
            if qpos != preqpos and preqpos != "":
                if qflag: qref.append(1)
                else: qref.append(0)
                qflag = False
            preqpos = qpos
            ref.append(l)
            if l == 1: qflag = True
        if qflag: qref.append(1)
        else: qref.append(0)

    preqpos = ""
    maxval = 0.0
    maxidx = -1
    with open(fpred, "rb") as f:
        for i, line in enumerate(f.readlines()):
            parts = line.strip().split()
            qpos, scr = parts[0], float(parts[4])
            if qpos != preqpos and preqpos != "":
                qpred_idx.append(maxidx)
                maxval = 0.0
                maxidx = -1
            preqpos = qpos
            if scr >= thre: pred.append(1)
            else: pred.append(0)
            if scr > maxval:
                maxidx = i
                maxval = scr
        qpred_idx.append(maxidx)

    match_cnt, ref_cnt, pred_cnt = 0.0, 0.0, 0.0
    for r, p in zip(ref, pred):
        if r == 1: ref_cnt += 1.0
        if p == 1: pred_cnt += 1.0
        if r == 1 and p == 1: match_cnt += 1.0
    prec, reca = match_cnt / pred_cnt, match_cnt / ref_cnt

    match_cnt, ref_cnt, pred_cnt = 0.0, 0.0, 0.0
    for r, pidx in zip(qref, qpred_idx):
        if r == 1: ref_cnt += 1.0
        if pred[pidx] >= thre: pred_cnt += 1.0
        if r == 1 and pred[pidx] >= thre and ref[pidx] == 1: match_cnt += 1.0
    qprec, qreca = match_cnt / pred_cnt, match_cnt / ref_cnt
    
    qmatch_cnt, qcnt = 0.0, 0.0
    for r, pidx in zip(qref, qpred_idx):
        qcnt += 1.0
        if r == 1 and pred[pidx] >= thre and ref[pidx] == 1: qmatch_cnt += 1.0
        elif r == 0 and pred[pidx] < thre: qmatch_cnt += 1.0
    qacc = qmatch_cnt / qcnt

    return [prec, reca, 2.0*prec*reca/(prec+reca), qprec, qreca, 2.0*qprec*qreca/(qprec+qreca), qacc]



if __name__ == "__main__":
    refname, predname = sys.argv[1], sys.argv[2]
    thre = 0.11
    if len(sys.argv) > 3:
        thre = float(sys.argv[3])
    results = get_prf(refname, predname, thre=thre)
	
    print "WikiQA Question Triggering: precision = %.4f, recall = %.4f, F1 = %.4f" %(results[3], results[4], results[5])

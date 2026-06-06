import numpy as np, pandas as pd
from src import config, data

# load target
df = data.load_train()
y = df[config.TARGET_COL].values
ids = df[config.ID_COL].values if hasattr(config,'ID_COL') else df['id'].values
order = pd.Series(np.arange(len(ids)), index=ids)

members = {
    'LGBM' : 'exp_034_lgbm_combined',
    'XGB'  : 'exp_043_xgb_freq3',
    'RealMLP':'exp_046_rmlp_nens24_full',
    'CatB' : 'exp_025_cat_yearcat',
    'TabICL':'exp_071_tabicl_raw_full',
}
P = {}
for name,f in members.items():
    o = pd.read_csv(f'experiments/oof/{f}.csv').set_index('id')['oof']
    P[name] = o.reindex(ids).values
P = pd.DataFrame(P, index=ids)
yv = y

names = list(members)
print("=== 개별 OOF AUC ===")
from sklearn.metrics import roc_auc_score
for n in names:
    print(f"  {n:8s} {roc_auc_score(yv,P[n]):.6f}")

# 1) prediction correlation (Pearson on raw probs)
pred_corr = P.corr()
# 2) residual correlation: residual = y - p  (signed error)
R = pd.DataFrame({n: yv - P[n].values for n in names}, index=ids)
resid_corr = R.corr()
# 3) rank-residual: AUC is rank-based -> use rank of pred, error = rank disagreement proxy
#    signed rank residual: (rank(p)/N) - y  approximates contribution to misranking
Rk = pd.DataFrame({n: pd.Series(P[n].values).rank().values/len(yv) - yv for n in names}, index=ids)
rank_resid_corr = Rk.corr()

def show(m, title):
    print(f"\n=== {title} ===")
    print("        " + " ".join(f"{n:>7s}" for n in names))
    for i in names:
        print(f"{i:8s}" + " ".join(f"{m.loc[i,j]:7.4f}" for j in names))

show(pred_corr, "예측 상관 (Pearson, raw prob)")
show(resid_corr, "오차 상관 (residual y-p)")
show(rank_resid_corr, "랭크-오차 상관 (AUC 관점)")

def offdiag_mean(m):
    vals=[m.loc[i,j] for i in names for j in names if i<j]
    return np.mean(vals), np.min(vals), np.max(vals)

pm=offdiag_mean(pred_corr); rm=offdiag_mean(resid_corr); km=offdiag_mean(rank_resid_corr)
print("\n=== 요약 (off-diagonal mean / min / max) ===")
print(f"  예측 상관     mean={pm[0]:.4f}  min={pm[1]:.4f}  max={pm[2]:.4f}")
print(f"  오차 상관     mean={rm[0]:.4f}  min={rm[1]:.4f}  max={rm[2]:.4f}")
print(f"  랭크-오차상관 mean={km[0]:.4f}  min={km[1]:.4f}  max={km[2]:.4f}")

# Bayes/irreducible probe: fraction of error mass in the borderline band
pbar = P.mean(axis=1).values
band = (pbar>0.2)&(pbar<0.8)
print(f"\n=== 경계영역(0.2<p̄<0.8) 진단 ===")
print(f"  경계 샘플 비율: {band.mean():.3f}")
err = np.abs(yv - pbar)
print(f"  전체 평균오차 |y-p̄|: {err.mean():.4f}")
print(f"  경계영역 평균오차  : {err[band].mean():.4f}")
print(f"  비경계 평균오차    : {err[~band].mean():.4f}")
# how much do members AGREE yet are WRONG (consensus errors = irreducible signature)
consensus_wrong = ((P.std(axis=1).values < 0.05) & (err>0.5))
print(f"  합의했는데 틀린 비율(std<0.05 & |y-p̄|>0.5): {consensus_wrong.mean():.4f}  (= 환원불가 후보)")

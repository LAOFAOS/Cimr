import numpy as np
import pandas as pd


def egfr_ckdepi_2021(creatinine, age, is_female, cr_unit="umol/L"):

    scr = np.asarray(creatinine, dtype=float)
    if cr_unit == "umol/L":
        scr = scr / 88.4
    scr = np.clip(scr, 0.1, None)   
    age = np.asarray(age, dtype=float)
    fem = np.asarray(is_female, dtype=float)

    kappa = np.where(fem == 1, 0.7, 0.9)
    alpha = np.where(fem == 1, -0.241, -0.302)
    ratio = scr / kappa
    egfr = (142.0
            * np.minimum(ratio, 1.0) ** alpha
            * np.maximum(ratio, 1.0) ** (-1.200)
            * 0.9938 ** age
            * np.where(fem == 1, 1.012, 1.0))
    return pd.Series(egfr, name="egfr")


def fib4(age, ast, alt, platelets):

    age = np.asarray(age, dtype=float)
    ast = np.asarray(ast, dtype=float)
    alt = np.asarray(alt, dtype=float)
    plt = np.asarray(platelets, dtype=float)
    alt_safe = np.where(alt <= 0, np.nan, alt)
    plt_safe = np.where(plt <= 0, np.nan, plt)
    val = (age * ast) / (plt_safe * np.sqrt(alt_safe))
    return pd.Series(val, name="fib4")


_CCI_WEIGHTS = {
    "mi": 1, "chf": 1, "pvd": 1, "cva": 1, "dementia": 1, "copd": 1,
    "rheum": 1, "pud": 1, "mild_liver": 1, "dm": 1,
    "dm_complicated": 2, "hemiplegia": 2, "renal": 2, "any_tumor": 2,
    "mod_severe_liver": 3, "metastatic": 6, "aids": 6,
}


def charlson_index(comorb_flags, age=None):

    score = None
    for key, flag in comorb_flags.items():
        w = _CCI_WEIGHTS.get(key)
        if w is None:
            continue
        f = pd.Series(np.asarray(flag, dtype=float)).fillna(0).clip(0, 1)
        contrib = f * w
        score = contrib if score is None else score + contrib
    if score is None:
        raise ValueError("111")
    if age is not None:
        age = np.asarray(age, dtype=float)
        age_pts = np.clip(np.floor((age - 40) / 10), 0, None)
        score = score + age_pts
    return pd.Series(np.asarray(score, dtype=float), name="charlson")


# ============================================================
# 人群 population —— 代谢/CVD 风险合成分 (透明可替换)
# ============================================================
def metabolic_risk_score(bmi=None, waist=None, sbp=None, dbp=None,
                         glucose=None, tg=None, tc=None, hdl_low=None,
                         smoking=None, sex_male=None, age=None):
    parts = []
    def add(cond, w=1.0):
        parts.append(pd.Series(np.asarray(cond, dtype=float)).fillna(0) * w)

    if waist is not None:   add(np.asarray(waist, float) >= 90)        
    elif bmi is not None:   add(np.asarray(bmi, float) >= 28)          
    if sbp is not None and dbp is not None:
        add((np.asarray(sbp, float) >= 130) | (np.asarray(dbp, float) >= 85))
    if glucose is not None: add(np.asarray(glucose, float) >= 5.6)    
    if tg is not None:      add(np.asarray(tg, float) >= 1.7)          
    if tc is not None:      add(np.asarray(tc, float) >= 5.2, 0.5)     
    if hdl_low is not None: add(np.asarray(hdl_low, float) == 1)       
    if smoking is not None: add(np.asarray(smoking, float) == 1, 1.0)  
    if sex_male is not None:add(np.asarray(sex_male, float) == 1, 0.5) 
    if age is not None:     add(np.asarray(age, float) >= 55, 1.0)    

    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return pd.Series(np.asarray(total, dtype=float), name="metabolic_risk")

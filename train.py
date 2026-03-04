import os
import argparse
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier, Pool

from prep import make_major_weight, prep_for_catboost, \
                apply_lecture_mode_encoding, add_scale_columns, add_cert_acq_count, \
                drop_high_missing_cols, fit_local_text_embedding, add_sentence_embedding_from_local_wv, \
                    reduce_embedding_dims_pca

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--target", type=str, default="completed")
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--f1_threshold", type=float, default=0.5)

    args = parser.parse_args()

    # ===== (추가) 재현성 시드 고정 =====
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ===== (추가) run 폴더 만들기: 실험날짜_시간_시드 =====
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # 예: 20260202_142355
    run_name = f"{ts}_seed{args.seed}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # (추가) 실험 설정 저장
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 전처리
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test_df  = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    # hit
    train_df = make_major_weight(train_df, out_col="major_weight")
    test_df  = make_major_weight(test_df,  out_col="major_weight")
    
    #  ======= ✅ 결측 95% 이상 컬럼 제거 (ID/target은 유지) ======
    # train_df, dropped_cols, _ = drop_high_missing_cols(
    #     train_df,
    #     threshold=0.95,
    #     keep=[args.id_col, args.target],
    #     treat_blank_as_na=True
    # )
    # test_df, _, _ = drop_high_missing_cols(
    #     test_df,
    #     threshold=0.95,
    #     keep=[args.id_col],   # test에는 target 없음
    #     treat_blank_as_na=True
    # )

    # print(f"[drop_high_missing_cols] dropped {len(dropped_cols)} cols:", dropped_cols)
    # # ==========================================================
    # 위 코드에서 나온 결과 대회 수상 경력 및 하단 2개 class3, class4 에서 결측치 많이 발생 
    # train_df = train_df.drop(columns=["class3", "class4"], errors="ignore")
    # test_df  = test_df.drop(columns=["class3", "class4"], errors="ignore")

    # === 강의 원하는 preference 분석 리스너 / 현직자 수 간 분리 하여 분석 === hit?
    train_df = add_scale_columns(train_df, col="incumbents_lecture_scale")
    test_df  = add_scale_columns(test_df,  col="incumbents_lecture_scale")
    # ===================================================================
    
    # # ============ 원하는 자격증 수 분석 ===================== << 똥
    # train_df = add_desired_cert_count(train_df)
    # test_df  = add_desired_cert_count(test_df)
    # # ======================================================

    # ============ 딴 자격증 수 분석 ===================== hit
    train_df = add_cert_acq_count(train_df)   # cert_acq_count 생성
    test_df  = add_cert_acq_count(test_df)
    # =========================================================
    
    train_df = train_df.drop(columns=["certificate_acquisition"], errors="ignore")
    test_df  = test_df.drop(columns=["certificate_acquisition"], errors="ignore")

    # =====================================================

    # train_df = add_lecture_interest_from_whyBDA_single(train_df)
    # test_df  = add_lecture_interest_from_whyBDA_single(test_df)

    # ================ 재등록 여부 측정 ==================== fail ????
    # train_df = map_re_registration(train_df)
    # test_df  = map_re_registration(test_df)
    
    # train_df = train_df.drop(columns=["re_registration"], errors="ignore")
    # test_df  = test_df.drop(columns=["re_registration"], errors="ignore")
    # ===================================================

    # # whyBDA 칼럼을 원핫 인코딩으로 변경 fail
    # train_df = add_whyBDA_onehot(train_df)
    # test_df  = add_whyBDA_onehot(test_df)
    
    # train_df = train_df.drop(columns=["whyBDA"], errors="ignore")
    # test_df  = test_df.drop(columns=["whyBDA"], errors="ignore")
    # # ========================

    train_df = apply_lecture_mode_encoding(train_df)
    test_df  = apply_lecture_mode_encoding(test_df)
    
    TEXT_COL = "incumbents_lecture_scale_reason"

    # 1) 제공 데이터(train+test) 텍스트만으로 임베딩 학습
    wv_model = fit_local_text_embedding(
        train_df, test_df,
        col=TEXT_COL,
        method="fasttext",      # 추천: fasttext (OOV에 강함)
        vector_size=64,
        window=5,
        min_count=2,
        seed=args.seed,
        epochs=30
    )

    # 2) 문장 벡터(평균 임베딩) 컬럼으로 붙이기
    train_df = add_sentence_embedding_from_local_wv(train_df, TEXT_COL, wv_model, prefix="reason_ft")
    test_df  = add_sentence_embedding_from_local_wv(test_df,  TEXT_COL, wv_model, prefix="reason_ft")
    
    train_df, test_df, pca = reduce_embedding_dims_pca(
    train_df, test_df,
    prefix="reason_ft",
    n_components=16,
    out_prefix="reason_pca",
    seed=args.seed,
    drop_original=False
)

    sub_df = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))

    X = train_df.drop(columns=[args.id_col, args.target])
    y = train_df[args.target].astype(int)
    X_test = test_df.drop(columns=[args.id_col])

    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    X_cb = prep_for_catboost(X, cat_cols)
    X_test_cb = prep_for_catboost(X_test, cat_cols)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    oof_proba = np.zeros(len(X_cb), dtype=float)
    test_proba_sum = np.zeros(len(X_test_cb), dtype=float)

    # (추가) fold별 메트릭 저장용
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cb, y), start=1):
        X_tr, y_tr = X_cb.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X_cb.iloc[va_idx], y.iloc[va_idx]

        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        scale_pos_weight = (neg / max(pos, 1))

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        
        # ============ 2026-02-03 데이터 증강 ====================
        # ✅ (증강) train fold만 오버샘플링
        # X_tr, y_tr = oversample_train_fold(
        #     X_tr, y_tr,
        #     seed=args.seed + fold,
        #     sampling_strategy="auto"   # 또는 0.7, 0.8 같은 숫자로 조절
        # )

        # # 증강 후 pos/neg 다시 계산 (가중치 계산에 쓰면 맞게 됨)
        # pos = (y_tr == 1).sum()
        # neg = (y_tr == 0).sum()
        # scale_pos_weight = (neg / max(pos, 1))

        # train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        
        # ===================== 데이터 증강 끝 ======= ===============
        valid_pool = Pool(X_va, y_va, cat_features=cat_cols)
        test_pool  = Pool(X_test_cb, cat_features=cat_cols)

        f1_metric = f"F1:proba_border={args.f1_threshold}"

        # fold마다 다른 seed 사용(지금 코드 유지)
        fold_seed = args.seed + fold

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric=f1_metric,
            iterations=5000,
            learning_rate=0.03,
            depth=8,
            random_seed=fold_seed,
            od_type="Iter",
            od_wait=200,
            scale_pos_weight=scale_pos_weight,
            verbose=200
        )

        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_proba = model.predict_proba(valid_pool)[:, 1]
        oof_proba[va_idx] = va_proba
        test_proba_sum += model.predict_proba(test_pool)[:, 1]

        va_pred = (va_proba >= args.f1_threshold).astype(int)
        acc = accuracy_score(y_va, va_pred)
        f1 = f1_score(y_va, va_pred)
        auc = roc_auc_score(y_va, va_proba)

        best_iter = model.get_best_iteration()
        print(f"[FOLD {fold}] ACC={acc:.4f} F1={f1:.4f} AUC={auc:.4f} | best_iter={best_iter}")

        # ===== (수정) 모델 저장 위치: run_dir =====
        model_path = os.path.join(run_dir, f"catboost_fold{fold}.cbm")
        model.save_model(model_path)

        # (추가) fold 메트릭 기록
        fold_rows.append({
            "fold": fold,
            "fold_seed": fold_seed,
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "best_iter": best_iter,
            "pos": int(pos),
            "neg": int(neg),
            "scale_pos_weight": float(scale_pos_weight),
        })

    # OOF
    oof_pred = (oof_proba >= args.f1_threshold).astype(int)
    oof_acc = accuracy_score(y, oof_pred)
    oof_f1  = f1_score(y, oof_pred)
    oof_auc = roc_auc_score(y, oof_proba)

    print("\n[OOF]")
    print(f"ACC: {oof_acc:.4f} | F1: {oof_f1:.4f} | AUC: {oof_auc:.4f}")

    # (추가) 메트릭 저장
    pd.DataFrame(fold_rows).to_csv(os.path.join(run_dir, "fold_metrics.csv"), index=False)
    pd.DataFrame([{
        "oof_acc": oof_acc,
        "oof_f1": oof_f1,
        "oof_auc": oof_auc,
        "f1_threshold": args.f1_threshold,
        "seed": args.seed,
        "n_splits": args.n_splits,
    }]).to_csv(os.path.join(run_dir, "oof_metrics.csv"), index=False)

    # test prediction
    test_proba = test_proba_sum / args.n_splits
    test_pred = (test_proba >= args.f1_threshold).astype(int)

    sub_df[args.target] = test_pred

    # ===== (수정) submission 저장 위치: run_dir =====
    out_sub_path = os.path.join(run_dir, "submission.csv")
    sub_df.to_csv(out_sub_path, index=False)
    print(f"Saved run -> {run_dir}")
    print(f"Saved submission -> {out_sub_path}")


if __name__ == "__main__":
    main()

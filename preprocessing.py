from __future__ import annotations
from typing import Iterable, Tuple, List, Optional

import re
import pandas as pd
import numpy as np



KEYWORDS = ["데이터", "소프트웨어", "컴퓨터", "통계", "인공지능", "AI", "디지털", "IT", "공학", "SW", "ai", "sw"]

def make_major_weight(
    df: pd.DataFrame,
    major1_col: str = "major1_1",
    major2_col: str = "major1_2",
    out_col: str = "major_weight",
    w_major1: int = 3,
    w_major2: int = 1,
    keywords=KEYWORDS
) -> pd.DataFrame:
    df = df.copy()

    # 결측/비문자 처리 대비해서 string으로 통일
    m1 = df[major1_col].astype("string").fillna("")
    m2 = df[major2_col].astype("string").fillna("")

    # 키워드 포함 여부 (부분 문자열 포함)
    pattern = "|".join(map(re.escape, keywords))
    hit1 = m1.str.contains(pattern, case=False, na=False)
    hit2 = m2.str.contains(pattern, case=False, na=False)

    # 1전공(3점) + 2전공(1점) 합산
    df[out_col] = hit1.astype(int) * w_major1 + hit2.astype(int) * w_major2

    return df


def prep_for_catboost(X: pd.DataFrame, cat_cols):
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")
    return X





# 온/ 오프라인 분류 

_MAP = {
    "온라인": 0,
    "오프라인": 1,
    "온,오프라인": 2,
    "온, 오프라인": 2,
}

def encode_lecture_mode(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(" ", "")
    return _MAP.get(s, pd.NA)

def apply_lecture_mode_encoding(df: pd.DataFrame,
                                cols=("hope_for_group", "incumbents_lecture_type")) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].apply(encode_lecture_mode).astype("Int64")
    return df


#  ===============  리스너, 현직자 수 분리 ========================
def parse_scale_text(s: str):
    """
    예시 입력:
      - "3~50명 내외의 강의 리스너와 1명의 현직자"
      - "100명 이상의 리스너와 10명 이상의 현직자"
    출력:
      (listener_min, listener_max, incumbent_min, incumbent_max)
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return (np.nan, np.nan, np.nan, np.nan)

    s = str(s).strip()

    # 리스너 텍스트 조각: "...리스너..." 앞부분을 우선 잡아보기
    # (문장이 항상 "리스너와" 형태면 이 방식이 잘 먹힘)
    listener_part = None
    m = re.search(r"(.+?)리스너", s)
    if m:
        listener_part = m.group(1)

    # 현직자 텍스트 조각
    incumbent_part = None
    m = re.search(r"현직자", s)
    if m:
        # "현직자" 앞에서 숫자 구간이 있는 조각을 잡기 위해 뒤에서부터 탐색
        # 간단히 "현직자" 앞 20~30글자 정도를 잘라서 파싱
        idx = s.find("현직자")
        incumbent_part = s[max(0, idx-40):idx]

    def parse_numbers(part: str):
        if not part:
            return (np.nan, np.nan)

        part = part.replace(" ", "")

        # 1) 범위: 3~50명, 3-50명
        m = re.search(r"(\d+)\s*[~\-]\s*(\d+)\s*명", part)
        if m:
            return (int(m.group(1)), int(m.group(2)))

        # 2) 이상: 100명 이상, 100명이상의
        m = re.search(r"(\d+)\s*명\s*이상", part)
        if m:
            return (int(m.group(1)), np.nan)

        # 3) 단일: 1명, 1명의
        m = re.search(r"(\d+)\s*명", part)
        if m:
            v = int(m.group(1))
            return (v, v)

        return (np.nan, np.nan)

    lmin, lmax = parse_numbers(listener_part)
    imin, imax = parse_numbers(incumbent_part)

    return (lmin, lmax, imin, imax)


def add_scale_columns(df: pd.DataFrame, col="certificate_acquisition"):
    df = df.copy()
    parsed = df[col].apply(parse_scale_text)

    df["listener_min"] = parsed.apply(lambda x: x[0])
    df["listener_max"] = parsed.apply(lambda x: x[1])
    df["incumbent_min"] = parsed.apply(lambda x: x[2])
    df["incumbent_max"] = parsed.apply(lambda x: x[3])

    # (선택) 모델에 그냥 한 숫자로 넣고 싶으면 대표값도 만들 수 있음
    # 범위면 중앙값, 이상이면 min값, 단일이면 그 값
    def representative(minv, maxv):
        if pd.isna(minv) and pd.isna(maxv):
            return np.nan
        if pd.isna(maxv):
            return float(minv)  # "이상"은 하한만
        return (float(minv) + float(maxv)) / 2.0

    df["listener_repr"] = df.apply(lambda r: representative(r["listener_min"], r["listener_max"]), axis=1)
    df["incumbent_repr"] = df.apply(lambda r: representative(r["incumbent_min"], r["incumbent_max"]), axis=1)

    return df

# ===============================================

# # ================== 원하는 자격증 수 정리 ============================ << 똥
# def add_desired_cert_count(df: pd.DataFrame, col="certificate_acquisition", out_col="desired_cert_count"):
#     df = df.copy()
#     s = df[col].astype("string").fillna("")

#     def count_items(x: str) -> int:
#         x = x.strip()
#         if x == "" or x.lower() in ["없음", "없어요", "없다", "none", "nan"]:
#             return 0
#         items = [t.strip() for t in x.split(",") if t.strip() != ""]
#         return len(items)

#     df[out_col] = s.apply(count_items).astype(int)
#     return df

# # ================================================

def add_cert_acq_count(
    df: pd.DataFrame,
    col: str = "certificate_acquisition",
    out_col: str = "cert_acq_count"
) -> pd.DataFrame:
    df = df.copy()
    s = df[col].astype("string").fillna("")

    def count_items(x: str) -> int:
        x = x.strip()
        if x == "" or x.lower() in ["없음", "없어요", "없다", "none", "nan"]:
            return 0
        items = [t.strip() for t in x.split(",") if t.strip() != ""]
        return len(items)

    df[out_col] = s.apply(count_items).astype(int)
    return df

# 똥
# def map_re_registration(df, col="re_registration", out_col=None):
#     df = df.copy()
#     if out_col is None:
#         out_col = col  # 원컬럼 덮어쓰기

#     mapping = {
#         "아니요": 0,
#         "예": 1,
#     }

#     s = df[col].astype("string").str.strip()
#     df[out_col] = s.map(mapping).astype("Int64")  # NaN 허용 정수형
#     return df

# ================== 결측치 삭제 ===================== << 대회 수상 경력 빼버리는데, 이건 좀 별로라
def drop_high_missing_cols(df: pd.DataFrame,
                           threshold: float = 0.95,
                           keep=None,
                           treat_blank_as_na: bool = True):
    """
    결측 비율(threshold 이상) 컬럼 삭제
    return: (df_out, dropped_cols, missing_ratio)
    """
    x = df.copy()

    if treat_blank_as_na:
        # "" / "   " 같은 값도 결측으로 처리
        x = x.replace(r"^\s*$", pd.NA, regex=True)

    missing_ratio = x.isna().mean()
    dropped_cols = missing_ratio[missing_ratio >= threshold].index.tolist()

    if keep:
        keep_set = set(keep)
        dropped_cols = [c for c in dropped_cols if c not in keep_set]

    return x.drop(columns=dropped_cols), dropped_cols, missing_ratio
# =============================================


def add_whyBDA_onehot(
    df: pd.DataFrame,
    col: str = "whyBDA",
    prefix: str = "whyBDA"
) -> pd.DataFrame:
    import re

    WHYBDA_ITEMS = {
        "01": "큰 규모인 만큼, 커리큘럼이나 운영 등 관리가 잘 될것 같아서",
        "02": "BDA 학회원만의 혜택을 누리고 싶어서",
        "03": "혼자 공부하기 어려워서",
        "04": "이전 기수에 매우 만족해서",
        "05": "시간적으로 부담이 없어서",
        "06": "학회 가입 시 코딩 테스트, 면접 등을 보지 않아서",
        "07": "현직자의 강의를 듣고 싶어서",
    }

    
    df = df.copy()
    s = df[col].astype("string").fillna("").str.strip()

    # 1) 번호 기반 매칭 (가장 신뢰)
    # ex) "01. ...\n02. ..." 안에 01. 02. 등이 있으면 True
    for code, text in WHYBDA_ITEMS.items():
        df[f"{prefix}_{code}"] = s.str.contains(rf"(?<!\d){re.escape(code)}\s*\.", regex=True).astype(int)

    # 2) 번호가 전혀 없는 경우(또는 일부만 있는 경우) 대비: 문장(키워드)로 백업 매칭
    # - 1)에서 잡힌 게 0인데, 텍스트가 포함되면 1로
    # - 너무 길면 변형이 많을 수 있어서 핵심 키워드로 줄이는 것도 가능
    for code, text in WHYBDA_ITEMS.items():
        colname = f"{prefix}_{code}"
        df[colname] = df[colname].where(df[colname] == 1, s.str.contains(re.escape(text), regex=True).astype(int))

    return df

#==================================================================

# 데이터 증강 << -- 쓸모없음 성능 개쳐낮아짐
# def oversample_train_fold(
#     X_tr: pd.DataFrame,
#     y_tr: pd.Series,
#     seed: int,
#     sampling_strategy="auto"
# ):
#     """
#     Train fold에서만 오버샘플링(복제)로 클래스 불균형 완화.
#     sampling_strategy:
#       - "auto": minority를 majority 수준까지 맞춤
#       - 0.7 : minority를 majority의 70% 수준까지 맞춤 (과적합 줄이기 좋음)
#     """
#     try:
#         from imblearn.over_sampling import RandomOverSampler
#     except ImportError as e:
#         raise ImportError(
#             "imblearn이 필요합니다. 설치: pip install imbalanced-learn"
#         ) from e

#     ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed)
#     X_res, y_res = ros.fit_resample(X_tr, y_tr)

#     # 혹시 numpy로 나오는 환경 대비
#     if not isinstance(X_res, pd.DataFrame):
#         X_res = pd.DataFrame(X_res, columns=X_tr.columns)

#     y_res = pd.Series(y_res).astype(int)
#     return X_res, y_res



import pandas as pd
import numpy as np

def add_lecture_interest_from_whyBDA_single(
    df: pd.DataFrame,
    col: str = "whyBDA",
    out_col: str = "lecture_interest"
) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df

    s = df[col].astype("string").fillna("").str.strip()

    # 너가 지정한 3개 이유면 "강의에 흥미"로 간주
    targets = {
        "혼자 공부하기 어려워서",
        "BDA 학회원만의 혜택을 누리고 싶어서(현직자 강연, 잡 페스티벌, 기업연계 공모전 등)",
        "현직자의 강의를 듣고 싶어서",
    }

    df[out_col] = s.isin(targets).astype(int)
    return df



# 의미없는 데이터 빼기 -> 똥
# def add_meaningless_text_score(
#     df: pd.DataFrame,
#     col: str,
#     out_col: str = None,
#     *,
#     penalty: int = -1,
#     ok_value: int = 0,
#     min_len: int = 2,
#     extra_bad_patterns: list[str] | None = None
# ) -> pd.DataFrame:
#     import re
    
#     """
#     col(텍스트)에서 '성의없는 답변'이면 penalty(-1) 점수 부여, 아니면 ok_value(0).
#     원본 텍스트는 유지하고 점수 피처만 추가하는 용도.
#     """
#     if col not in df.columns:
#         return df

#     out = df.copy()
#     if out_col is None:
#         out_col = f"{col}_score"

#     s = out[col].astype("string")

#     bad_patterns = [
#         r"^\s*$",
#         r"^(없음|무응답|응답\s*없음|해당\s*없음|해당없음|없어요|없습니다)$",
#         r"^(모르겠|잘\s*모르겠|모름|기억\s*안|기억안|몰라)$",
#         r"^(x|X|-\s*|—|_+|\.+|/)$",
#         r"^(n/?a|N/?A|none|None|NO|no)$",
#         r"^(그냥|글쎄|음|ㅇㅇ|ㄴㄴ|ㅇㅋ|ㄱㅅ|감사|감사합니다)$",
#         r"^(ㅋ+|ㅎ+|ㅠ+|ㅜ+|ㅜㅜ+|ㅠㅠ+|;;+|:?\)+|:?\(+)$",
#     ]
#     if extra_bad_patterns:
#         bad_patterns += extra_bad_patterns

#     bad_re = re.compile("|".join(f"(?:{p})" for p in bad_patterns), flags=re.IGNORECASE)

#     s2 = s.fillna("").str.strip()
#     compact = s2.str.replace(r"\s+", "", regex=True)

#     is_bad = s2.apply(lambda x: bool(bad_re.match(x)))
#     too_short = compact.str.len() < int(min_len)

#     meaningless = is_bad | too_short

#     out[out_col] = ok_value
#     out.loc[meaningless, out_col] = penalty
#     return out


import re
import numpy as np
import pandas as pd

from gensim.models import Word2Vec, FastText

def _simple_tokenize_ko(text: str) -> list[str]:
    """
    외부 형태소 분석기 없이(=외부데이터/사전 없이) 쓰는 초간단 토크나이저.
    - 한글/영문/숫자만 남기고 나머지 제거
    - 공백 기준 split
    """
    text = str(text)
    text = text.strip()
    if not text:
        return []
    # 한글/영문/숫자/공백만 남기기
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return text.split(" ")

def fit_local_text_embedding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    *,
    method: str = "fasttext",   # "word2vec" or "fasttext"
    vector_size: int = 64,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    seed: int = 42,
    sg: int = 1,                # 1=skip-gram, 0=cbow
    epochs: int = 20
):
    """
    제공 데이터(train+test)의 col 텍스트만으로 Word2Vec/FastText 학습.
    반환: 학습된 gensim 모델
    """
    # train + test 텍스트 합쳐서(외부 데이터 없이) 학습
    all_texts = pd.concat([train_df[col], test_df[col]], axis=0, ignore_index=True)
    sentences = [ _simple_tokenize_ko(t) for t in all_texts.fillna("").astype(str).tolist() ]
    sentences = [s for s in sentences if len(s) > 0]

    if method.lower() == "word2vec":
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            seed=seed
        )
    elif method.lower() == "fasttext":
        model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            seed=seed
        )
    else:
        raise ValueError("method must be 'word2vec' or 'fasttext'")

    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model

def add_sentence_embedding_from_local_wv(
    df: pd.DataFrame,
    col: str,
    model,
    *,
    prefix: str = "reason_wv",
    normalize: bool = True
) -> pd.DataFrame:
    """
    학습된 word vectors(model.wv)로 문장 벡터 생성:
    문장 벡터 = (문장 내 단어 벡터들의 평균)
    - FastText는 OOV(학습에 없던 단어)도 subword로 벡터 생성 가능(장점)
    """
    out = df.copy()
    wv = model.wv
    dim = wv.vector_size

    texts = out[col].fillna("").astype(str).tolist()
    sent_vecs = np.zeros((len(texts), dim), dtype=np.float32)

    for i, t in enumerate(texts):
        tokens = _simple_tokenize_ko(t)
        vecs = []
        for tok in tokens:
            # FastText는 대부분 tok in wv 체크 없이도 벡터를 주지만,
            # Word2Vec은 없는 단어면 KeyError -> 안전하게 처리
            if tok in wv:
                vecs.append(wv[tok])
            else:
                # FastText면 wv[tok] 시도해도 됨(보통 생성됨). 안전하게 try.
                try:
                    vecs.append(wv[tok])
                except Exception:
                    pass

        if vecs:
            v = np.mean(vecs, axis=0)
            if normalize:
                n = np.linalg.norm(v) + 1e-12
                v = v / n
            sent_vecs[i] = v
        # vecs 없으면 0벡터 유지

    emb_df = pd.DataFrame(sent_vecs, columns=[f"{prefix}_{j}" for j in range(dim)])
    out = pd.concat([out.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    return out

from sklearn.decomposition import PCA

def reduce_embedding_dims_pca(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    prefix: str,              # 예: "reason_ft"
    n_components: int = 16,   # 줄일 차원
    out_prefix: str = None,   # 예: "reason_pca"
    seed: int = 42,
    drop_original: bool = True
):
    """
    train_df/test_df에 있는 prefix_0..prefix_{d-1} 임베딩 컬럼을 PCA로 차원 축소.
    - PCA는 train_df로만 fit (누수 방지)
    - train/test 모두 transform 적용
    반환: (train_df_new, test_df_new, fitted_pca)
    """
    if out_prefix is None:
        out_prefix = f"{prefix}_pca"

    # 임베딩 컬럼 찾기
    emb_cols = [c for c in train_df.columns if c.startswith(prefix + "_")]
    if len(emb_cols) == 0:
        return train_df, test_df, None

    X_tr = train_df[emb_cols].to_numpy(dtype=np.float32)
    X_te = test_df[emb_cols].to_numpy(dtype=np.float32)

    # PCA fit on train only
    pca = PCA(n_components=n_components, random_state=seed)
    Z_tr = pca.fit_transform(X_tr)
    Z_te = pca.transform(X_te)

    # 새 컬럼 붙이기
    train_new = train_df.copy()
    test_new = test_df.copy()

    pca_cols = [f"{out_prefix}_{i}" for i in range(n_components)]
    train_new[pca_cols] = Z_tr
    test_new[pca_cols] = Z_te

    # 원본 임베딩 컬럼 제거(원하면)
    if drop_original:
        train_new = train_new.drop(columns=emb_cols, errors="ignore")
        test_new = test_new.drop(columns=emb_cols, errors="ignore")

    return train_new, test_new, pca

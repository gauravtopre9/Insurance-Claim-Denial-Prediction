
from sklearn.preprocessing import LabelEncoder
from src.data_loader import root_dir
from src.logger import setup_logger
# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)



def fit_encoding_config(hist_fe, config):
    train_raw  = hist_fe[hist_fe["split"] == "train"].copy()
    SMOOTH     = 10.0
    glob_mean  = train_raw["is_denied"].mean()
    CAT_COLS = config['col_details']['CAT_COLS']
    CUST_COLS =config['col_details']['CUST_COL']
    DROP = config['col_details']['DROP']

    return train_raw, SMOOTH, glob_mean, CAT_COLS, CUST_COLS, DROP


def fit_label_encoders(train_raw, CAT_COLS, CUST_COLS):

    label_encs = {}
    for col in CAT_COLS+CUST_COLS:
        le = LabelEncoder().fit(train_raw[col].astype(str))
        label_encs[col] = le
        logger.info(f"  Label  {col:12s} → {len(le.classes_)} classes")
    return label_encs


def fit_target_encoders(train_raw, CAT_COLS, CUST_COLS, SMOOTH, glob_mean):

    te_maps = {}
    for col in CAT_COLS+CUST_COLS:
        agg = train_raw.groupby(col)["is_denied"].agg(["sum","count"])
        agg["te"] = (agg["sum"] + SMOOTH * glob_mean) / (agg["count"] + SMOOTH)
        te_maps[col] = agg["te"].to_dict()
        logger.info(f"  Target {col:12s} : min={agg['te'].min():.3f}  max={agg['te'].max():.3f}")
    return te_maps


def encode_df(df, CAT_COLS, CUST_COLS, te_maps, label_encs, glob_mean):
    d = df.copy()
    for col in CAT_COLS+CUST_COLS:
        d[f"te_{col}"] = d[col].astype(str).map(te_maps[col]).fillna(glob_mean)

    for col in CAT_COLS+CUST_COLS:
        le   = label_encs[col]
        d[col] = d[col].astype(str).apply(lambda x: x if x in set(le.classes_) else le.classes_[0])
        d[col] = le.transform(d[col])
    return d


def build_feature_columns(hist_enc, cur_enc, DROP):
    FEAT_COLS = [c for c in hist_enc.columns if c not in (DROP or ["is_denied"])]


    for col in FEAT_COLS:
        if col not in cur_enc.columns:
            cur_enc[col] = 0

    X_train = hist_enc[hist_enc["split"]=="train"][FEAT_COLS].values
    y_train = hist_enc[hist_enc["split"]=="train"]["is_denied"].values
    X_val   = hist_enc[hist_enc["split"]=="validation"][FEAT_COLS].values
    y_val   = hist_enc[hist_enc["split"]=="validation"]["is_denied"].values
    X_test  = hist_enc[hist_enc["split"]=="test"][FEAT_COLS].values
    y_test  = hist_enc[hist_enc["split"]=="test"]["is_denied"].values
    X_cur   = cur_enc[FEAT_COLS].values

    return FEAT_COLS, X_train, y_train, X_val, y_val, X_test, y_test, X_cur


def run_encoding(hist_fe, cur_fe, config):
    train_raw, SMOOTH, glob_mean, CAT_COLS, CUST_COLS, DROP = fit_encoding_config(hist_fe, config)

    label_encs = fit_label_encoders(train_raw, CAT_COLS, CUST_COLS)
    te_maps = fit_target_encoders(train_raw, CAT_COLS, CUST_COLS, SMOOTH, glob_mean)

    hist_enc = encode_df(hist_fe, CAT_COLS, CUST_COLS, te_maps, label_encs, glob_mean)
    cur_enc  = encode_df(cur_fe, CAT_COLS, CUST_COLS, te_maps, label_encs, glob_mean)

    FEAT_COLS, X_train, y_train, X_val, y_val, X_test, y_test, X_cur = build_feature_columns(hist_enc, cur_enc, DROP)

    logger.info(cur_enc)  

    return {
        "train_raw": train_raw, "SMOOTH": SMOOTH, "glob_mean": glob_mean,
        "CAT_COLS": CAT_COLS, "CUST_COLS": CUST_COLS, "DROP": DROP,
        "label_encs": label_encs, "te_maps": te_maps,
        "hist_enc": hist_enc, "cur_enc": cur_enc,
        "FEAT_COLS": FEAT_COLS,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_cur": X_cur,
    }

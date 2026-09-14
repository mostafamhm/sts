import os
import sys
import time
import logging
import signal
import gc
import argparse
import clickhouse_connect
import numpy as np
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] REEMBED_WORKER: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ReembedWorker")


DB_HOST = os.getenv("CLICKHOUSE_HOST", '172.20.70.191')
DB_PORT = int(os.getenv("CLICKHOUSE_PORT", 8123))
DB_USER = os.getenv("CLICKHOUSE_USER", 'labafi')
DB_PASS = os.getenv("CLICKHOUSE_PASS", 'l@b@fi@1234')


SOURCE_DB = os.getenv("SOURCE_DB", "raya_sepehr_preprocessed")
SOURCE_TABLE = f"{SOURCE_DB}.embeddings"

COL_PLATFORM = "platform"
COL_ROWKEY = "row_key"
COL_EMBED_NEW = "embedding_2"
COL_EMBED_NEW_VERSION = "embedding_2_version"
EMBED_VERSION_VALUE = "bge3_finetuned"

MAX_FETCH_SIZE = int(os.getenv("MAX_FETCH_SIZE", 500))
GPU_BATCH_SIZE = int(os.getenv("GPU_BATCH_SIZE", 128))
IDLE_WAIT_TIME = int(os.getenv("IDLE_WAIT_TIME", 20))
ERROR_BACKOFF_TIME = int(os.getenv("ERROR_BACKOFF_TIME", 10))


PLATFORM_CONFIG = {
    "x": {
        "ref_db": "x",
        "ref_table": "tweets_2",
        "key_cols": ["user_id", "tweet_id"],
        "text_col": "txtContent",
        "num_parts": 2,
    },
    "eita": {
        "ref_db": "eita",
        "ref_table": "posts",
        "key_cols": ["channel", "msgid"],
        "text_col": "txtContent",
        "num_parts": 2,
    },
    "bale": {
        "ref_db": "bale",
        "ref_table": "posts",
        "key_cols": ["channel_id", "msgid"],
        "text_col": "txtContent",
        "num_parts": 2,
    },
    "telegram": {
        "ref_db": "telegram",
        "posts_table": "posts",
        "comments_table": "comments",
        "num_parts": 3,   # channel || msgid || comment_id
    },
}

stop_requested = False

def signal_handler(sig, frame):
    global stop_requested
    logger.info("🛑 Stop signal received. Finishing current batch...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    from utils import similarity_model
    from preprocessing import dynamic_preprocess, PreprocessingOptions

    logger.info("🧠 Loading Model & Warming up...")
    warmup_emb = similarity_model.encode_texts(["warmup"])

    if hasattr(warmup_emb, 'shape'):
        EMBEDDING_DIM = warmup_emb.shape[1]
    else:
        EMBEDDING_DIM = len(warmup_emb[0])

    ZERO_VECTOR = [0.0] * EMBEDDING_DIM

    cleaner_opts = PreprocessingOptions(
        url_replacement="", mention_replacement="", hashtag_mode="clean",
        emoji_mode="remove", remove_special_chars=True,
        normalize_whitespace=True
    )
    logger.info(f"✅ Model Ready. Dim: {EMBEDDING_DIM}")
except ImportError:
    logger.error("❌ Dependencies (utils/preprocessing) not found.")
    sys.exit(1)


def get_client():
    return clickhouse_connect.get_client(
        host=DB_HOST, port=DB_PORT, username=DB_USER, password=DB_PASS
    )


def sql_escape(value) -> str:

    return str(value).replace("\\", "\\\\").replace("'", "\\'")

def sql_str(value) -> str:
    return f"'{sql_escape(value)}'"

def sql_tuple_list(tuples) -> str:

    parts = []
    for t in tuples:
        inner = ",".join(sql_str(v) for v in t)
        parts.append(f"({inner})")
    return "(" + ",".join(parts) + ")"



def fetch_pending(client, platform, limit):
    query = f"""
    SELECT {COL_ROWKEY}
    FROM {SOURCE_TABLE}
    WHERE {COL_PLATFORM} = {sql_str(platform)}
      AND (empty({COL_EMBED_NEW}) OR {COL_EMBED_NEW} IS NULL)
      AND (empty({COL_EMBED_NEW_VERSION}) OR {COL_EMBED_NEW_VERSION} IS NULL)
    LIMIT {limit}
    """
    return client.query_df(query)

def parse_row_keys(platform, row_keys):
    expected_parts = PLATFORM_CONFIG[platform]["num_parts"]
    parsed = []
    for rk in row_keys:
        parts = str(rk).split("||")
        if len(parts) != expected_parts:
            logger.warning(f"⚠️ row_key نامعتبر (تعداد پارت‌ها={len(parts)}، انتظار={expected_parts}): {rk}")
            continue
        parsed.append({"row_key": rk, "parts": parts})
    return parsed

def fetch_reference_texts(client, platform, parsed_rows):
    """از دیتابیس مرجع همون پلتفرم، متن اصلی پیام‌ها رو واکشی می‌کنه -> dict[row_key] = txtContent"""
    if not parsed_rows:
        return {}

    if platform == "telegram":
        cfg = PLATFORM_CONFIG["telegram"]
        triples = [tuple(r["parts"]) for r in parsed_rows]
        # posts و comments بر اساس channel+msgid مرج می‌شن؛ چون date/txtContent
        # تو هر دو جدول هست، فقط ستون‌های لازم رو با alias صریح انتخاب می‌کنیم
        # تا تداخل اسم پیش نیاد. متنی که امبد می‌شه txtContent خود comment هست.
        query = f"""
        SELECT
            toString(c.channel)    AS channel,
            toString(c.msgid)      AS msgid,
            toString(c.comment_id) AS comment_id,
            c.txtContent           AS txtContent
        FROM {cfg['ref_db']}.{cfg['comments_table']} AS c
        INNER JOIN {cfg['ref_db']}.{cfg['posts_table']} AS p
            ON toString(c.channel) = toString(p.channel)
           AND toString(c.msgid)   = toString(p.msgid)
        WHERE (toString(c.channel), toString(c.msgid), toString(c.comment_id))
              IN {sql_tuple_list(triples)}
        """
        df = client.query_df(query)
        text_map = {}
        for _, row in df.iterrows():
            rk = f"{row['channel']}||{row['msgid']}||{row['comment_id']}"
            text_map[rk] = row["txtContent"]
        return text_map

    cfg = PLATFORM_CONFIG[platform]
    k1, k2 = cfg["key_cols"]
    pairs = [tuple(r["parts"]) for r in parsed_rows]
    query = f"""
    SELECT toString({k1}) AS k1, toString({k2}) AS k2, {cfg['text_col']} AS txtContent
    FROM {cfg['ref_db']}.{cfg['ref_table']}
    WHERE (toString({k1}), toString({k2})) IN {sql_tuple_list(pairs)}
    """
    df = client.query_df(query)
    text_map = {}
    for _, row in df.iterrows():
        rk = f"{row['k1']}||{row['k2']}"
        text_map[rk] = row["txtContent"]
    return text_map



def update_embedding(client, platform, row_key, vector):
    client.command(
        f"""
        ALTER TABLE {SOURCE_TABLE}
        UPDATE {COL_EMBED_NEW} = {{emb:Array(Float32)}},
               {COL_EMBED_NEW_VERSION} = {{ver:String}}
        WHERE {COL_PLATFORM} = {{plat:String}} AND {COL_ROWKEY} = {{rk:String}}
        """,
        parameters={"emb": vector, "ver": EMBED_VERSION_VALUE, "plat": platform, "rk": row_key},
        settings={"mutations_sync": 1},
    )



# def run_stream_worker(platform):
#     client = get_client()
#     logger.info(f"🚀 Re-embed Worker Started for platform='{platform}'")
#     total_processed = 0

#     while not stop_requested:
#         cycle_start = time.time()
#         try:
#             df = fetch_pending(client, platform, MAX_FETCH_SIZE)

#             if df.empty:
#                 logger.info(f"💤 No new data. Sleeping {IDLE_WAIT_TIME}s...")
#                 time.sleep(IDLE_WAIT_TIME)
#                 continue

#             parsed_rows = parse_row_keys(platform, df[COL_ROWKEY].tolist())
#             if not parsed_rows:
#                 continue

#             # ۱. واکشی متن اصلی از جدول مرجع پلتفرم
#             text_map = fetch_reference_texts(client, platform, parsed_rows)

#             valid_texts, valid_keys = [], []
#             zero_vector_keys = []

#             for r in parsed_rows:
#                 row_key = r["row_key"]
#                 raw_txt = text_map.get(row_key)

#                 if raw_txt is None or str(raw_txt).lower() in ['na', 'nan'] or not isinstance(raw_txt, str):
#                     zero_vector_keys.append(row_key)
#                     continue

#                 txt = dynamic_preprocess(raw_txt, cleaner_opts)
#                 if not txt or len(txt.strip()) < 2:
#                     zero_vector_keys.append(row_key)
#                 else:
#                     valid_texts.append(txt)
#                     valid_keys.append(row_key)

#             for row_key in zero_vector_keys:
#                 update_embedding(client, platform, row_key, ZERO_VECTOR)
#             for i in range(0, len(valid_texts), GPU_BATCH_SIZE):
#                 if stop_requested:
#                     break
#                 sub_texts = valid_texts[i:i + GPU_BATCH_SIZE]
#                 sub_keys = valid_keys[i:i + GPU_BATCH_SIZE]
#                 try:
#                     embs = similarity_model.encode_texts(sub_texts)
#                     if hasattr(embs, 'cpu'):
#                         embs = embs.cpu().numpy()
#                     for rk, emb in zip(sub_keys, embs):
#                         update_embedding(client, platform, rk, emb.tolist())
#                 except Exception as e:
#                     logger.error(f"❌ GPU Error in batch: {e}")
#                     for rk in sub_keys:
#                         update_embedding(client, platform, rk, ZERO_VECTOR)

#             total_processed += len(parsed_rows)
#             duration = time.time() - cycle_start
#             logger.info(
#                 f"✨ Cycle done: {len(parsed_rows)} rows | Total: {total_processed} | "
#                 f"Speed: {len(parsed_rows)/duration:.1f} r/s"
#             )

#             del df, parsed_rows, text_map
#             gc.collect()

#         except Exception as e:
#             logger.error(f"💥 Critical Stream Error: {e}")
#             time.sleep(ERROR_BACKOFF_TIME)
#             try:
#                 client = get_client()
#             except Exception:
#                 pass

#     logger.info("🏁 Re-embed Worker stopped.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Re-embed messages with the new (v2) model per platform.")
#     parser.add_argument(
#         "platform",
#         choices=list(PLATFORM_CONFIG.keys()),
#         help="پلتفرم مورد پردازش: x | telegram | bale | eita",
#     )
#     args = parser.parse_args()
#     run_stream_worker(args.platform)



def run_stream_worker():
    client = get_client()
    logger.info("🚀 Re-embed Worker Started for all platforms")
    platforms = list(PLATFORM_CONFIG.keys())

    while not stop_requested:
        data_found_in_any_platform = False

        for platform in platforms:
            if stop_requested:
                break

            logger.info(f"Checking platform: {platform}")
            platform_processed = 0
            while not stop_requested:
                cycle_start = time.time()
                try:
                    df = fetch_pending(client, platform, MAX_FETCH_SIZE)

                    if df.empty:
                        logger.info(f"💤 No new data for {platform}. Moving to next platform...")
                        break 

                    data_found_in_any_platform = True
                    parsed_rows = parse_row_keys(platform, df[COL_ROWKEY].tolist())
                    if not parsed_rows:
                        continue

                    text_map = fetch_reference_texts(client, platform, parsed_rows)

                    valid_texts, valid_keys = [], []
                    zero_vector_keys = []

                    for r in parsed_rows:
                        row_key = r["row_key"]
                        raw_txt = text_map.get(row_key)

                        if raw_txt is None or str(raw_txt).lower() in ['na', 'nan'] or not isinstance(raw_txt, str):
                            zero_vector_keys.append(row_key)
                            continue

                        txt = dynamic_preprocess(raw_txt, cleaner_opts)
                        if not txt or len(txt.strip()) < 2:
                            zero_vector_keys.append(row_key)
                        else:
                            valid_texts.append(txt)
                            valid_keys.append(row_key)

                    for row_key in zero_vector_keys:
                        update_embedding(client, platform, row_key, ZERO_VECTOR)
                    
                    for i in range(0, len(valid_texts), GPU_BATCH_SIZE):
                        if stop_requested:
                            break
                        sub_texts = valid_texts[i:i + GPU_BATCH_SIZE]
                        sub_keys = valid_keys[i:i + GPU_BATCH_SIZE]
                        try:
                            embs = similarity_model.encode_texts(sub_texts)
                            if hasattr(embs, 'cpu'):
                                embs = embs.cpu().numpy()
                            for rk, emb in zip(sub_keys, embs):
                                update_embedding(client, platform, rk, emb.tolist())
                        except Exception as e:
                            logger.error(f"GPU Error in batch: {e}")
                            for rk in sub_keys:
                                update_embedding(client, platform, rk, ZERO_VECTOR)

                    platform_processed += len(parsed_rows)
                    duration = time.time() - cycle_start
                    logger.info(
                        f"Cycle done for {platform}: {len(parsed_rows)} rows | Platform Total: {platform_processed} | "
                        f"Speed: {len(parsed_rows)/duration:.1f} r/s"
                    )

                    del df, parsed_rows, text_map
                    gc.collect()

                except Exception as e:
                    logger.error(f"Critical Stream Error for {platform}: {e}")
                    time.sleep(ERROR_BACKOFF_TIME)
                    try:
                        client = get_client()
                    except Exception:
                        pass
                    break # در صورت بروز خطای دیتابیس، از این پلتفرم خارج شده و بعدی را تست می‌کند

  
        if not data_found_in_any_platform and not stop_requested:
            logger.info(f"💤 No data in ANY platform. Sleeping {IDLE_WAIT_TIME}s before next global check...")
            time.sleep(IDLE_WAIT_TIME)

    logger.info("🏁 Re-embed Worker stopped.")


if __name__ == "__main__":
    run_stream_worker()

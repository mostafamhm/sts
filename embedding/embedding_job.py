import os
import time
import logging
import signal
import gc
import clickhouse_connect
import numpy as np
from dotenv import load_dotenv

# ---------------- CONFIGURATION ----------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] STREAM_WORKER: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("StreamEmbedder")

DB_HOST = os.getenv("CLICKHOUSE_HOST", '172.20.70.191')
DB_PORT = int(os.getenv("CLICKHOUSE_PORT", 8123))
DB_NAME = os.getenv("CLICKHOUSE_DB", 'telegram')
DB_USER = os.getenv("CLICKHOUSE_USER", 'labafi')
DB_PASS = os.getenv("CLICKHOUSE_PASS", 'l@b@fi@1234')

MAIN_TABLE = f"{DB_NAME}.posts"
STAGE_TABLE = f"{DB_NAME}.posts_updates_stage"
COL_MSG_ID = "msgid"
COL_CHAN_NAME = "channel"
COL_TEXT = "txtContent"
COL_EMBED = "embedding_bge"

MAX_FETCH_SIZE = 5000     # برای استریم سایز کوچکتر بهتر است
GPU_BATCH_SIZE = 128 
IDLE_WAIT_TIME = 20      # زمان انتظار وقتی داده جدید نیست
ERROR_BACKOFF_TIME = 10 

stop_requested = False

def signal_handler(sig, frame):
    global stop_requested
    logger.info("🛑 Stop signal received. Finishing current batch...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------------- MODEL & PREPROCESSING ----------------
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
    exit(1)

# ---------------- DATABASE ----------------

def get_client():
    return clickhouse_connect.get_client(
        host=DB_HOST, port=DB_PORT, username=DB_USER, password=DB_PASS
    )

# ---------------- STREAM LOGIC ----------------

def run_stream_worker():
    client = get_client()
    
    logger.info(f"🚀 Stream Worker Started (Insert-Only Mode)")
    total_processed = 0

    while not stop_requested:
        cycle_start = time.time()
        try:
            fetch_query = f"""
            SELECT {COL_CHAN_NAME}, {COL_MSG_ID}, {COL_TEXT}
            FROM {MAIN_TABLE}
            WHERE (empty({COL_EMBED}) OR {COL_EMBED} IS NULL)
              AND ({COL_CHAN_NAME}, {COL_MSG_ID}) NOT IN (SELECT channel_name, msgid FROM {STAGE_TABLE})
            LIMIT {MAX_FETCH_SIZE}
            """
            
            df = client.query_df(fetch_query)
            
            if df.empty:
                logger.info(f"💤 No new data. Sleeping {IDLE_WAIT_TIME}s...")
                time.sleep(IDLE_WAIT_TIME)
                continue
            
            valid_texts = []
            valid_meta = []
            insert_data = []

            # ۲. Preprocessing
            for _, row in df.iterrows():
                raw_txt = row[COL_TEXT]
                
                # فیلتر کردن مقادیر پوچ یا غیررشته‌ای (مشابه کد Worker)
                if raw_txt is None or str(raw_txt).lower() in ['na', 'nan'] or not isinstance(raw_txt, str):
                    insert_data.append([row[COL_CHAN_NAME], row[COL_MSG_ID], ZERO_VECTOR])
                    continue
                
                txt = dynamic_preprocess(raw_txt, cleaner_opts)
                if not txt or len(txt.strip()) < 2:
                    insert_data.append([row[COL_CHAN_NAME], row[COL_MSG_ID], ZERO_VECTOR])
                else:
                    valid_texts.append(txt)
                    valid_meta.append((row[COL_CHAN_NAME], row[COL_MSG_ID]))

            # ۳. Embedding (GPU Batching)
            if valid_texts:
                for i in range(0, len(valid_texts), GPU_BATCH_SIZE):
                    if stop_requested: break
                    sub_batch = valid_texts[i : i + GPU_BATCH_SIZE]
                    try:
                        embs = similarity_model.encode_texts(sub_batch)
                        if hasattr(embs, 'cpu'): embs = embs.cpu().numpy()
                        
                        batch_meta = valid_meta[i : i + GPU_BATCH_SIZE]
                        for (c_name, m_id), emb in zip(batch_meta, embs):
                            insert_data.append([c_name, m_id, emb.tolist()])
                    except Exception as e:
                        logger.error(f"❌ GPU Error in batch: {e}")
                        # در صورت خطا، برای حفظ جریان، بردار صفر درج می‌کنیم
                        for (c_name, m_id) in valid_meta[i : i + GPU_BATCH_SIZE]:
                            insert_data.append([c_name, m_id, ZERO_VECTOR])

            # ۴. Bulk Insert به جدول استیج
            if insert_data:
                client.insert(
                    STAGE_TABLE,
                    insert_data,
                    column_names=['channel_name', 'msgid', 'embedding_bge']
                )
                
                total_processed += len(insert_data)
                duration = time.time() - cycle_start
                logger.info(f"✨ Stream Batch: {len(insert_data)} rows | Total: {total_processed} | Speed: {len(insert_data)/duration:.1f} r/s")

            # تمیزکاری حافظه
            del df, insert_data, valid_texts, valid_meta
            gc.collect()

        except Exception as e:
            logger.error(f"💥 Critical Stream Error: {e}")
            time.sleep(ERROR_BACKOFF_TIME)
            # بازسازی اتصال در صورت خرابی
            try: client = get_client()
            except: pass

    logger.info(f"🏁 Stream Worker stopped. Data remains in {STAGE_TABLE}")

if __name__ == "__main__":
    run_stream_worker()
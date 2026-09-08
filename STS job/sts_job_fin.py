import os
import time
import logging
import argparse
import datetime
import pg8000
from clickhouse_connect import get_client
from dotenv import load_dotenv

load_dotenv()

DIM = 1024
ID_BUFFER_FILE = "processed_msgids_buffer.txt"
RESULTS_TXT_FILE = "results_output.txt"
UPDATE_INTERVAL_SECONDS = 300 
UPDATE_BATCH_THRESHOLD = 5000 


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

parser = argparse.ArgumentParser(description="Similarity Search Job")
parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)', default=None)
parser.add_argument('--topic_id', type=int, help='Specific Topic ID', default=None)
parser.add_argument('--social_id', type=int, help='Social Network ID (Default 2 for Telegram)', default=2)
parser.add_argument('--top_k', type=int, help='Top K results', default=10)
parser.add_argument('--min_score', type=float, help='Min Similarity Score', default=0.0)
parser.add_argument('--max_score', type=float, help='Max Similarity Score', default=100.0)

args = parser.parse_args()

TOP_K = args.top_k
MIN_SCORE = args.min_score
MAX_SCORE = args.max_score
SOCIAL_ID = args.social_id


TOTAL_PROCESSED_COUNT = 0

def get_pg_conn():
    return pg8000.connect(
        host=os.getenv("PG_HOST", '172.20.70.191'),
        port=int(os.getenv("PG_PORT", 5432)),
        database=os.getenv("PG_DB", 'olap'),
        user=os.getenv("PG_USER", 'labafi'),
        password=os.getenv("PG_PASS", 'l@b@fi@1234')
    )

def get_ch_client():
    return get_client(
        host=os.getenv("CH_HOST", '172.20.70.191'),
        port=int(os.getenv("CH_PORT", 8123)),
        database=CH_DB_NAME,
        username=os.getenv("CH_USER", 'labafi'),
        password=os.getenv("CH_PASS", 'l@b@fi@1234')
    )
    
def get_social_info(social_id):
    """استخراج نام دیتابیس کلیک‌هاوس بر اساس ID از جدول پلتفرم‌ها"""
    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT en_label FROM socials WHERE id = %s", (social_id,))
        result = cur.fetchone()
        return result[0] if result else "telegram"
    finally:
        conn.close()
        
CH_DB_NAME = get_social_info(SOCIAL_ID)
logging.info(f"🔍 Targeting Social ID: {SOCIAL_ID} | Database: {CH_DB_NAME}")

        
def append_id_to_file(msg_id):
    try:
        with open(ID_BUFFER_FILE, "a") as f:
            f.write(f"{msg_id}\n")
    except Exception as e:
        logging.error(f"❌ File Append Error: {e}")

def read_ids_from_file():
    if not os.path.exists(ID_BUFFER_FILE): return []
    try:
        with open(ID_BUFFER_FILE, "r") as f:
            return [int(line.strip()) for line in f if line.strip().isdigit()]
    except Exception as e:
        logging.error(f"❌ File Read Error: {e}")
        return []

def clear_file():
    try:
        with open(ID_BUFFER_FILE, "w") as f: f.truncate(0)
    except Exception as e:
        logging.error(f"❌ File Clear Error: {e}")



def ch_array_str(strings):
    if not strings: return "[]"
    escaped = [s.replace("'", "\\'") for s in strings]
    return "[" + ",".join(f"'{s}'" for s in escaped) + "]"

def fetch_topics_map(target_topic_id=None):
    topic_map = {}
    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        query = "SELECT id FROM topics"
        if target_topic_id:
            query += " WHERE id = %s"
            cur.execute(query, (target_topic_id,))
        else:
            cur.execute(query)
        
        tids = [r[0] for r in cur.fetchall()]
        for tid in tids:
            cur.execute("SELECT keyword FROM topic_keywords WHERE topic_id = %s", (tid,))
            kws = [r[0] for r in cur.fetchall()]
            if kws: topic_map[tid] = kws
    finally:
        conn.close()
    return topic_map

def find_similar_in_ch(query_id, query_vec, keywords, ch_client, candidate_date_sql):
    kws_sql = ch_array_str(keywords)
    
    query_sql = f"""
    SELECT 
        p.msgid, 
        p.txtContent, 
        dotProduct(u.embedding_bge, %(qvec)s) as score
    FROM posts AS p
    INNER JOIN posts_updates_stage AS u ON p.msgid = u.msgid AND p.channel = u.channel_name
    WHERE p.date >= {candidate_date_sql}
      AND p.msgid != %(qid)s
      AND multiSearchAny(p.txtContent, {kws_sql})
      AND length(u.embedding_bge) = {DIM}
    HAVING score >= %(min_s)s AND score <= %(max_s)s
    ORDER BY score DESC 
    LIMIT %(k)s
    """
    return ch_client.query(query_sql, parameters={
        "qvec": query_vec, "qid": query_id, "k": TOP_K,
        "min_s": MIN_SCORE, "max_s": MAX_SCORE
    })
    

def flush_flags_to_ch(ch_client):
    ids = read_ids_from_file()
    if not ids: return 0
    ids_str = ",".join(str(i) for i in ids)
    try:
        # آپدیت ستون فلگ در جدول اصلی
        ch_client.command(f"ALTER TABLE posts UPDATE embedding_bge = [1.0] WHERE msgid IN ({ids_str})")
        clear_file()
        return len(ids)
    except Exception as e:
        logging.error(f"❌ Flush Error: {e}")
        return 0

def save_results_to_pg(buffer):
    """
    Returns:
        bool: True if successful, False otherwise
    """
    if not buffer: return True
    conn = get_pg_conn()
    try:
        cur = conn.cursor()
        insert_query = """
            INSERT INTO similarity_results (
                query_message_id, similar_message_id, query_message, 
                similar_message, topic_id, score
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (query_message_id, similar_message_id) DO NOTHING
        """
        # اضافه کردن SOCIAL_ID به بافر برای ثبت در دیتابیس
        
        cur.executemany(insert_query, buffer)
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"❌ PostgreSQL Sync Error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()
        
        
def run_job():
    global TOTAL_PROCESSED_COUNT
    ch_client = get_ch_client()
    # بازه زمانی برای جستجوی پیام‌های مشابه
    candidate_date_sql = f"toDate('{args.start_date}')" if args.start_date else "now() - INTERVAL 900 DAY"
    
    logging.info(f"🚀 STS Production Job Started. Mode: Continuous Loop.")
    
    if os.path.exists(ID_BUFFER_FILE):
        flush_flags_to_ch(ch_client)

    last_flush_time = time.time()

    while True:
        try:
            # واکشی آخرین لیست تاپیک‌ها
            topic_map = fetch_topics_map(args.topic_id)
            if not topic_map:
                logging.info("No topics to process. Sleeping...")
                time.sleep(30)
                continue

            cycle_activity = False
            
            for topic_id, keywords in topic_map.items():
                kws_sql = ch_array_str(keywords)
                
                # پیدا کردن پیام‌های جدید
                hunter_query = f"""
                SELECT p.msgid, p.txtContent, u.embedding_bge
                FROM posts AS p
                INNER JOIN posts_updates_stage AS u ON p.msgid = u.msgid AND p.channel = u.channel_name
                WHERE p.date >= now() - INTERVAL 180 DAY
                  AND length(p.embedding_bge) = 0 
                  AND length(u.embedding_bge) = {DIM}
                  AND multiSearchAny(p.txtContent, {kws_sql})
                LIMIT 50
                """
                
                new_msgs = ch_client.query(hunter_query)
                if not new_msgs.result_rows:
                    continue

                cycle_activity = True
                pg_buffer = []
                # لیست موقت برای نگهداری IDها تا قبل از ذخیره موفق
                batch_pending_ids = []
                
                for row in new_msgs.result_rows:
                    qid, qtext, qvec = row
                    
                    start_time = time.time()
                    
                    # جستجوی شباهت در دیتابیس برای این پیام
                    sim_res = find_similar_in_ch(qid, qvec, keywords, ch_client, candidate_date_sql)
                    
                    duration = time.time() - start_time

                    logging.info(f"⏱️  [QID: {qid}] Found {len(sim_res.result_rows)} matches in {duration:.3f}s")
                    
                    for srow in sim_res.result_rows:
                        sid, stext, score = srow
                        pg_buffer.append((qid, sid, qtext, stext, topic_id, float(score)))
                    
                    # اضافه کردن به لیست انتظار برای فلگ زدن
                    batch_pending_ids.append(qid)


                    TOTAL_PROCESSED_COUNT += 1
                    if TOTAL_PROCESSED_COUNT % 5 == 0:
                         logging.info(f"📊 Progress: {TOTAL_PROCESSED_COUNT} items processed so far.")

                # ذخیره دسته‌ای نتایج در دیتابیس (حالا اول انجام می‌شود)
                if pg_buffer:
                    t_save_start = time.time()
                    success = save_results_to_pg(pg_buffer)
                    t_save_end = time.time()
                    
                    if success:
                        logging.info(f"✅ Batch Saved to PG in {t_save_end - t_save_start:.3f}s. Marking {len(batch_pending_ids)} IDs as processed.")
                        # حالا که ذخیره موفق بود، IDها را به فایل اضافه کن
                        for processed_id in batch_pending_ids:
                            append_id_to_file(processed_id)
                    else:
                        logging.error("❌ Failed to save batch to PG. NOT marking IDs as processed to retry later.")
                elif batch_pending_ids:
                    # حالتی که نتایجی برای شباهت پیدا نشده اما خود پیام پردازش شده است
                    # باید پیام را فلگ بزنیم تا دوباره پردازش نشود
                    logging.info(f"⚠️ No similarities found for batch, but marking {len(batch_pending_ids)} IDs as processed.")
                    for processed_id in batch_pending_ids:
                        append_id_to_file(processed_id)


            current_pending = read_ids_from_file()
            time_since_flush = time.time() - last_flush_time
            
            if len(current_pending) >= UPDATE_BATCH_THRESHOLD or (len(current_pending) > 0 and time_since_flush >= UPDATE_INTERVAL_SECONDS):
                count = flush_flags_to_ch(ch_client)
                logging.info(f"✨ Flushed {count} processed IDs to ClickHouse.")
                last_flush_time = time.time()

            # مدیریت وقفه
            if not cycle_activity:
                time.sleep(15)
            else:
                time.sleep(1)

        except KeyboardInterrupt:
            logging.info("Stopping job...")
            flush_flags_to_ch(ch_client)
            break
        except Exception as e:
            logging.error(f"💥 Critical Error: {e}", exc_info=True)
            time.sleep(10)

if __name__ == "__main__":
    run_job()
























































# ### اگر نیاز به تست بود میتوانید کد زیر را اجرا کنید که بصورت محدود شباهت سنجی انجام داده و نتایج را در فایل تکست ذخیره میکند.

# # def run_job():
# #     ch_client = get_ch_client()
# #     # در تست، اگر تاریخ ندادیم بازه رو بازتر می‌گیریم تا حتماً دیتا پیدا بشه
# #     candidate_date_sql = f"toDate('{args.start_date}')" if args.start_date else "now() - INTERVAL 800 DAY"
    
# #     logging.info(f"🚀 [TEST MODE] STS Job Started. Goal: 20 target msgs per topic.")
    
# #     topic_results_count = {} # شمارش پیام‌های هدف (Query Messages) پیدا شده
    
# #     # واکشی تاپیک‌ها یکبار در ابتدای تست
# #     topic_map = fetch_topics_map(args.topic_id)
# #     if not topic_map:
# #         logging.error("❌ No topics found. Exiting...")
# #         return

# #     # حلقه اصلی تست
# #     while True:
# #         all_topics_done = True
# #         cycle_activity = False
        
# #         for topic_id, keywords in topic_map.items():
# #             # چک کردن حد نصاب ۲۰ پیام هدف برای هر تاپیک
# #             current_count = topic_results_count.get(topic_id, 0)
# #             if current_count >= MAX_RESULTS_PER_TOPIC:
# #                 continue
            
# #             all_topics_done = False # هنوز حداقل یک تاپیک تمام نشده
# #             kws_sql = ch_array_str(keywords)
            
# #             # هانتر کوئری: واکشی پیام‌های باقی‌مانده تا رسیدن به ۲۰
# #             needed = MAX_RESULTS_PER_TOPIC - current_count
# #             hunter_query = f"""
# #             SELECT p.msgid, p.txtContent, u.embedding_bge
# #             FROM posts AS p
# #             INNER JOIN posts_updates_stage AS u ON p.msgid = u.msgid AND p.channel = u.channel_name
# #             WHERE p.date >= now() - INTERVAL 800 DAY
# #               AND length(p.embedding_bge) = 0 
# #               AND length(u.embedding_bge) = {DIM}
# #               AND multiSearchAny(p.txtContent, {kws_sql})
# #             LIMIT {needed}
# #             """
            
# #             new_msgs = ch_client.query(hunter_query)
# #             if not new_msgs.result_rows:
# #                 continue

# #             cycle_activity = True
# #             pg_buffer = []
            
# #             for row in new_msgs.result_rows:
# #                 qid, qtext, qvec = row
                
# #                 # پیدا کردن ۱۰ مورد مشابه (TOP_K) برای این پیام هدف
# #                 sim_res = find_similar_in_ch(qid, qvec, keywords, ch_client, candidate_date_sql)
                
# #                 for srow in sim_res.result_rows:
# #                     sid, stext, score = srow
# #                     pg_buffer.append((qid, sid, qtext, stext, topic_id, float(score)))
                
# #                 # ثبت آیدی برای فلگ زدن (که در تست هم انجام بشه تا دیتای تکراری نگیریم)
# #                 append_id_to_file(qid)
# #                 topic_results_count[topic_id] = topic_results_count.get(topic_id, 0) + 1

# #             if pg_buffer:
# #                 save_to_txt(pg_buffer)
# #                 logging.info(f"✅ Topic {topic_id}: Found {len(new_msgs.result_rows)} new targets. Total: {topic_results_count[topic_id]}/20")

# #         # شرط خروج از تست
# #         if all_topics_done:
# #             logging.info("🎯 [TEST COMPLETED] All topics reached 20 target messages. Exiting...")
# #             flush_flags_to_ch(ch_client) # انجام آخرین آپدیت فلگ‌ها
# #             break
        
# #         if not cycle_activity:
# #             logging.warning("⚠️ No more data found for remaining topics. Exiting to avoid infinite loop...")
# #             flush_flags_to_ch(ch_client)
# #             break
        
# #         time.sleep(1)

# # if __name__ == "__main__":
# #     run_job()
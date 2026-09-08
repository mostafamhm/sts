import re
import emoji
from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator
from parsivar import Normalizer


parsivar_normalizer = Normalizer()

class PreprocessingOptions(BaseModel):
    url_replacement: Optional[str] = Field(
        default="[URL]", 
        description="رشته‌ای برای جایگزینی URLها. اگر null ارسال شود، URLها دست‌نخورده باقی می‌مانند."
        )
    mention_replacement: Optional[str] = Field(
        default="[USER]", 
        description="رشته‌ای برای جایگزینی منشن‌ها. اگر null ارسال شود، منشن‌ها دست‌نخورده باقی می‌مانند."
        )
    hashtag_mode: Literal["clean", "remove", "keep"] = Field(
        default="clean", 
        description="نحوه برخورد با هشتگ‌ها: clean (پاکسازی)، remove (حذف کامل)، keep (دست‌نخورده)."
        )
    emoji_mode: Literal["demojize", "remove", "keep"] = Field(
        default="demojize", 
        description="نحوه برخورد با ایموجی‌ها: demojize (تبدیل به متن)، remove (حذف)، keep (دست‌نخورده)."
        )
    remove_special_chars: bool = Field(
        default=True, 
        description="فعال/غیرفعال کردن حذف کاراکترهای خاص بر اساس الگوی پیش‌فرض."
        )
    allowed_chars: Optional[str] = Field(
        default=None, 
        description="رشته‌ای از کاراکترهای خاص که باید در متن نگه داشته شوند (علاوه بر حروف و اعداد)."
        )
    custom_regex_pattern: Optional[str] = Field(
        default=None, 
        description="یک الگوی Regex سفارشی برای حذف کاراکترها. در صورت استفاده، دو گزینه قبلی نادیده گرفته می‌شوند."
        )
    number_normalization: Optional[Literal["to_persian", "to_english"]] = Field(
        default=None, 
        description="تبدیل اعداد به فارسی (to_persian) یا انگلیسی (to_english)."
        )
    character_normalization: bool = Field(
        default=False,
        description="نرمال‌سازی حروف عربی به فارسی (مانند ك -> ک) با استفاده از Parsivar."
    )
    normalize_whitespace: bool = Field(
        default=True, 
        description="فعال/غیرفعال کردن نرمال‌سازی فاصله‌ها."
        )

    @model_validator(mode='before')
    def check_special_chars_options(cls, data: dict) -> dict:
        if data.get('custom_regex_pattern') and data.get('allowed_chars'):
            raise ValueError("نمی‌توان همزمان از 'custom_regex_pattern' و 'allowed_chars' استفاده کرد.")
        return data

def _handle_urls(text: str, replacement: str) -> str:
    return re.sub(r'http\S+|www\S+', replacement, text)

def _handle_mentions(text: str, replacement: str) -> str:
    return re.sub(r'@\w+', replacement, text)

def _handle_hashtags(text: str, mode: str) -> str:
    if mode == "clean":
        return re.sub(r'#(\S+)', lambda m: m.group(1).replace('_', ' '), text)
    if mode == "remove":
        return re.sub(r'#\S+', '', text)
    return text

def _handle_emojis(text: str, mode: str) -> str:
    if mode == "demojize":
        return emoji.demojize(text, delimiters=(" ", " "))
    if mode == "remove":
        return emoji.replace_emoji(text, replace='')
    return text


def _handle_special_chars(text: str, remove: bool, allowed: Optional[str], custom_pattern: Optional[str]) -> str:
    if custom_pattern:
        return re.sub(custom_pattern, '', text)
    if not remove:
        return text
    base_pattern = r'\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9\s.?!'
    if allowed:
        allowed_safe = re.escape(allowed)
        final_pattern = f'[^{base_pattern}{allowed_safe}]'
    else:
        final_pattern = f'[^{base_pattern}]'
    return re.sub(final_pattern, '', text)

def _normalize_numbers(text: str, mode: Optional[str]) -> str:
    if mode is None: return text
    persian_to_english_map = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
    english_to_persian_map = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
    if mode == "to_english": return text.translate(persian_to_english_map)
    if mode == "to_persian": return text.translate(english_to_persian_map)
    return text

def _normalize_characters(text: str, normalize: bool) -> str:
    if not normalize: return text
    return parsivar_normalizer.normalize(text)

def _normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def dynamic_preprocess(text: str, options: PreprocessingOptions) -> str:
    processed_text = text
    processed_text = _normalize_characters(processed_text, options.character_normalization)
    processed_text = _normalize_numbers(processed_text, options.number_normalization)
    if options.url_replacement is not None:
        processed_text = _handle_urls(processed_text, options.url_replacement)
    if options.mention_replacement is not None:
        processed_text = _handle_mentions(processed_text, options.mention_replacement)
    processed_text = _handle_hashtags(processed_text, options.hashtag_mode)
    processed_text = _handle_emojis(processed_text, options.emoji_mode)
    processed_text = _handle_special_chars(
        processed_text, 
        options.remove_special_chars, 
        options.allowed_chars, 
        options.custom_regex_pattern
    )
    if options.normalize_whitespace:
        processed_text = _normalize_whitespace(processed_text)
    if not processed_text:
        return '[None]'
    return processed_text

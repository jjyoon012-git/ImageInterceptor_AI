"""
dataset_collector_youtube.py
키워드 기반 YouTube CC 동영상 썸네일 수집 (harmful / safe_hard / safe_easy)
비율: harmful : safe_hard : safe_easy = 1 : 2 : 0.5
"""

import os
import time
import json
import random
import requests
from pathlib import Path
from urllib.parse import urlencode
from PIL import Image
from io import BytesIO

# === YouTube API 설정 ===
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "Google Cloud Console에서 발급받은 키 입력하면 됩니다.")
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

# 카테고리별 키워드
KEYWORDS = {
    #"harmful": ["유해 키워드 입력"],
     #"safe_hard": ["비유해하지만 유해 오인 가능 키워드 입력"],
    # "safe_easy": ["비유해 키워드 입력"]
}

# 저장 경로
SAVE_ROOT = Path("data_collected_youtube")
SAVE_ROOT.mkdir(exist_ok=True)

# 비율
TARGET_RATIO = {"safe_hard": 0, "safe_easy": 2}
BASE_COUNT = 200  # harmful 200이면 safe_hard 400, safe_easy 100
TARGET_COUNTS = {k: int(v * BASE_COUNT) for k, v in TARGET_RATIO.items()}

# 기타 옵션
RESULTS_PER_PAGE = 50     # max 50
MAX_PAGES_PER_QUERY = 10  # 키워드당 최대 페이지
REQUEST_SLEEP = 0.10      # rate-limit 여유 (초)

def yt_search_ids(query, page_token=None):
    """YouTube 검색으로 videoId 리스트 가져오기 (CC만)"""
    params = {
        "key": YOUTUBE_API_KEY,
        "q": query,
        "part": "snippet",
        "type": "video",
        "videoLicense": "creativeCommon",   # CC-BY 만
        "maxResults": RESULTS_PER_PAGE,
        "safeSearch": "none",               # harmful 수집을 위해 none. 필요시 'moderate/strict'
        "relevanceLanguage": "ko",          # 필요시 조정/삭제
        # "regionCode": "US",               # 필요시 지정
        "pageToken": page_token or ""
    }
    url = f"{SEARCH_URL}?{urlencode(params)}"
    resp = requests.get(url, timeout=15).json()
    items = resp.get("items", [])
    ids = [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]
    next_token = resp.get("nextPageToken")
    time.sleep(REQUEST_SLEEP)
    return ids, next_token

def yt_videos_meta(video_ids):
    """video API로 상세 메타(채널명, 라이선스, 썸네일 등) 조회"""
    if not video_ids:
        return []
    params = {
        "key": YOUTUBE_API_KEY,
        "id": ",".join(video_ids),
        "part": "snippet,contentDetails,status"
    }
    url = f"{VIDEOS_URL}?{urlencode(params)}"
    resp = requests.get(url, timeout=15).json()
    items = resp.get("items", [])
    time.sleep(REQUEST_SLEEP)
    metas = []
    for it in items:
        snip = it.get("snippet", {})
        thumbs = snip.get("thumbnails", {}) or {}
        # 가장 큰 썸네일 우선(high -> standard -> maxres도 있을 수 있음)
        pick = thumbs.get("maxres") or thumbs.get("standard") or thumbs.get("high") \
               or thumbs.get("medium") or thumbs.get("default")
        metas.append({
            "videoId": it.get("id"),
            "title": snip.get("title"),
            "channelTitle": snip.get("channelTitle"),
            "publishedAt": snip.get("publishedAt"),
            "license": it.get("status", {}).get("license"),  # creativeCommon or youtube
            "thumbnail": pick.get("url") if pick else None
        })
    return metas

def download_thumbnail(url, save_dir, idx):
    """썸네일 다운로드 후 저장"""
    if not url:
        return False
    try:
        r = requests.get(url, timeout=15)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{idx:05d}.jpg"
        img.save(save_path, quality=92)
        return True, str(save_path)
    except Exception as e:
        print(f"썸네일 다운로드 실패: {url} -> {e}")
        return False, None

def fetch_one_category(cat, keywords, target_count):
    print(f"[{cat}] 목표 {target_count}장 수집...")
    save_dir = SAVE_ROOT / cat
    count = 0

    manifest_path = Path("metadata")
    manifest_path.mkdir(exist_ok=True)
    mf = open(manifest_path / f"youtube_manifest_{cat}.jsonl", "w", encoding="utf-8")

    try:
        while count < target_count:
            kw = random.choice(keywords)
            page_token = None
            pages = 0
            while count < target_count and pages < MAX_PAGES_PER_QUERY:
                # 1) 검색 -> videoId들
                ids, page_token = yt_search_ids(kw, page_token)
                if not ids:
                    break
                # 2) 상세 메타
                metas = yt_videos_meta(ids)
                for m in metas:
                    if m.get("license") != "creativeCommon":
                        continue
                    ok, path = download_thumbnail(m.get("thumbnail"), save_dir, count)
                    if not ok:
                        continue

                    record = {
                        "category": cat,
                        "keyword": kw,
                        "path": path,
                        "source": "youtube",
                        "videoId": m["videoId"],
                        "title": m["title"],
                        "channel": m["channelTitle"],
                        "publishedAt": m["publishedAt"],
                        "license": m["license"],
                        "thumbnail_url": m["thumbnail"],
                    }
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= target_count:
                        break

                pages += 1
                if not page_token:
                    break
        print(f"[{cat}] 완료: {count}장 저장됨")
    finally:
        mf.close()

if __name__ == "__main__":
    # 폴더 준비
    (Path("metadata")).mkdir(exist_ok=True)
    for cat, kws in KEYWORDS.items():
        fetch_one_category(cat, kws, TARGET_COUNTS[cat])
    print("✅ 전체 수집 완료")

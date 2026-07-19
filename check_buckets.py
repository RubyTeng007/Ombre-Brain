import asyncio
from bucket_manager import BucketManager
from utils import load_config

async def main():
    config = load_config()
    bm = BucketManager(config)
    buckets = await bm.list_all(include_archive=True)
    
    print(f"Total buckets: {len(buckets)}")
    
    domains = {}
    for b in buckets:
        for d in b.get("metadata", {}).get("domain", []):
            domains[d] = domains.get(d, 0) + 1
            
    print(f"Domains: {domains}")
    
    # Check for formatting issues (e.g., missing critical fields)
    issues = 0
    for b in buckets:
        meta = b.get("metadata", {})
        # feel/mirage 的 domain=[] 是設計（B-10）不是格式錯誤（audit F5）——
        # 116 個假陽性會把真訊號淹掉。
        domain_ok = bool(meta.get("domain")) or meta.get("type") in ("feel", "mirage")
        if not meta.get("name") or not domain_ok or not b.get("content"):
            print(f"Format issue in {b['id']}")
            issues += 1
            
    print(f"Found {issues} formatting issues.")

if __name__ == "__main__":
    asyncio.run(main())

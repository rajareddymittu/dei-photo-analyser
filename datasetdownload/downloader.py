import os
from icrawler.builtin import BingImageCrawler

TARGET_DIR = "./dataset_1000_photos_"
os.makedirs(TARGET_DIR, exist_ok=True)

# 10 categories across sportswear brands and global attire (100 photos each = 1,000 photos)
SEARCH_QUERIES = [
    # Sportswear Brands (Natural / Streetwear / Candid)
    "person wearing nike hoodie street candid",
    "candid person adidas tracksuit street",
    "person wearing puma sportswear outdoor",
    "runner wearing new balance sneakers city street",
    "person wearing under armour sportswear outdoor",
    
    # Global Attire (Natural / Candid Environments)
    "candid street photo woman wearing saree market",
    "candid person wearing kimono street kyoto",
    "candid street photo man wearing dashiki accra",
    "candid person wearing poncho cusco street",
    "candid person wearing bavarian dirndl outdoor"
]

TOTAL_PER_QUERY = 100

for idx, query in enumerate(SEARCH_QUERIES):
    subfolder = os.path.join(TARGET_DIR, f"category_{idx+1}_{query.split()[2]}")
    os.makedirs(subfolder, exist_ok=True)
    
    print(f"\n[{idx+1}/{len(SEARCH_QUERIES)}] Downloading: '{query}'...")
    
    crawler = BingImageCrawler(
        downloader_threads=4,
        storage={'root_dir': subfolder}
    )
    
    # Filter for full-sized photos (large / high resolution)
    crawler.crawl(
        keyword=query,
        filters=dict(size='large', type='photo'),
        max_num=TOTAL_PER_QUERY,
        min_size=(720, 720)
    )

print(f"\nDownload complete. All images stored in {TARGET_DIR}.")
"""JD 网页抓取：用 Crawl4AI 把招聘页转成 LLM 友好的 Markdown。

针对 SPA（字节/腾讯等招聘站）的处理：
- 增加渲染等待时间，让 JS 把职位详情渲染出来
- 滚动到底部触发懒加载内容
"""
import asyncio

try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
except ImportError as e:
    raise ImportError(
        "Crawl4AI 未安装。本地运行请：pip install -r requirements-local.txt && playwright install chromium"
    ) from e


async def _scrape(url: str) -> str:
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        delay_before_return_html=3.0,
        scan_full_page=True,
        wait_until="networkidle",
        page_timeout=30000,
    )
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url, config=config)
        return result.markdown or ""


def scrape_url(url: str) -> str:
    """同步接口：URL → Markdown 文本。"""
    return asyncio.run(_scrape(url))


if __name__ == "__main__":
    import sys
    print(scrape_url(sys.argv[1]))

import os
import asyncio
import json
from datetime import datetime
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from pydantic import BaseModel

class HikingEvent(BaseModel):
    title: str
    date: str
    location: str
    description: str
    link: str
    language: str

class HikingScraper:
    def __init__(self):
        self.deepseek_client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    async def extract_hiking_events(self, markdown: str, url: str) -> List[HikingEvent]:
        """Extract hiking events from markdown content using DeepSeek v3"""
        system_prompt = """You are an expert at extracting hiking event information from website content.
        Extract upcoming hiking events including title, date, location, description and link.
        Only extract events that are in the future.

        For descriptions:
        - Always try to extract a meaningful description
        - If no explicit description is found, create a brief summary from available event details
        - Description should be at least 20 characters long
        - Include key details like difficulty level, duration, or special requirements if available

        CRITICAL URL INSTRUCTIONS:
        - Extract URLs EXACTLY as they appear in the source content
        - DO NOT modify URLs in any way
        - DO NOT add any path segments (like /category/ or /product/)
        - DO NOT add angle brackets or any other characters
        - If you modify URLs, the links will break and the application will fail
        - This is the most important instruction - URLs must remain unchanged

        Return the results in JSON format with the following fields:
        - title: Event title
        - date: Event date in YYYY-MM-DD format
        - location: Event location, must not be empty
        - description: Event description or summary (minimum 20 characters)
        - link: The EXACT URL from the source content, unchanged in any way
        - language: Detect and specify the language as Serbian (sr), Croatian (hr), Slovenian (sl), or English (en) without translating to English.

        The response should be a JSON object with an 'events' array containing these fields."""

        try:
            response = await self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": markdown}
                ],
                response_format={"type": "json_object"}
            )

            raw_response = response.choices[0].message.content
            print(f"Raw API response: {raw_response}")
            response_data = json.loads(raw_response)
            print(f"Parsed events data: {response_data}")
            events_data = response_data.get("events", [])

            # Validate and filter events with empty or too short descriptions
            valid_events = []
            for event in events_data:
                if not event.get("description") or len(event["description"].strip()) < 20:
                    print(f"Skipping event '{event.get('title')}' due to missing or too short description")
                    continue

                # Clean URL if needed
                url = event.get("link", "")
                if url:
                    # Remove unwanted /category/ segment
                    if "/category/" in url:
                        url = url.replace("/category", "")
                    # Remove angle brackets if present
                    if "</" in url:
                        url = url.replace("</", "").replace(">", "")
                    event["link"] = url

                valid_events.append(event)

            return [HikingEvent(**event) for event in valid_events]
        except Exception as e:
            print(f"Error extracting events: {e}")
            return []

    async def crawl_website(self, url: str) -> List[HikingEvent]:
        """Crawl a single website and extract hiking events"""
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()

        try:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id="hiking_scraper"
            )

            if result.success:
                return await self.extract_hiking_events(result.markdown_v2.raw_markdown, url)
            return []
        finally:
            await crawler.close()

    async def crawl_websites(self, urls: List[str], max_concurrent: int = 5) -> List[HikingEvent]:
        """Crawl multiple websites in parallel"""
        semaphore = asyncio.Semaphore(max_concurrent)
        all_events = []

        async def process_url(url: str):
            async with semaphore:
                events = await self.crawl_website(url)
                all_events.extend(events)

        await asyncio.gather(*[process_url(url) for url in urls])
        return all_events

    def format_events_markdown(self, events: List[HikingEvent]) -> str:
        """Format hiking events as markdown"""
        markdown = "# Upcoming Hiking Events\n\n"

        # Group by language
        events_by_lang = {}
        for event in events:
            if event.language not in events_by_lang:
                events_by_lang[event.language] = []
            events_by_lang[event.language].append(event)

        for lang, lang_events in events_by_lang.items():
            markdown += f"## {lang.upper()} Events\n\n"
            for event in lang_events:
                markdown += f"### {event.title}\n"
                markdown += f"- **Date**: {event.date}\n"
                markdown += f"- **Location**: {event.location}\n"
                markdown += f"- **Description**: {event.description}\n"
                markdown += f"- [More Info]({event.link})\n\n"

        return markdown

async def main():
    # Load websites from config
    with open("hiking_config.json") as f:
        config = json.load(f)

    scraper = HikingScraper()
    events = await scraper.crawl_websites(config["websites"])

    # Debug print events
    print("Extracted events:")
    for event in events:
        print(event)

    # Save results
    markdown = scraper.format_events_markdown(events)
    with open("hiking_events.md", "w") as f:
        f.write(markdown)

    print(f"Found {len(events)} upcoming hiking events")

if __name__ == "__main__":
    asyncio.run(main())
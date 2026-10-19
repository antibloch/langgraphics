"""Real ReAct agent with live news and weather tools; no additional API keys.

uv run --group react python examples/news_weather_react_ifm.py
"""

import argparse
import asyncio
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from openai import AsyncOpenAI
from three_tool_react_ifm import build_agent

from langgraphics import watch


def fetch(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "LanggraphicsDemo/1.0"}
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        return response.read(2_000_000)


@tool
def check_news(query: str, limit: int = 5) -> dict:
    """Fetch current news headlines for a topic from Google News RSS.

    Returns titles, publication dates, publishers and source links, not article bodies.
    """
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query + " when:7d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    root = ET.fromstring(fetch(url))
    items = [
        {
            "title": item.findtext("title"),
            "published": item.findtext("pubDate"),
            "publisher": item.findtext("source"),
            "url": item.findtext("link"),
        }
        for item in root.findall("./channel/item")[: max(1, min(limit, 10))]
    ]
    return {
        "query": query,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "headlines": items,
        "scope": "RSS headlines only; article contents not verified",
    }


@tool
def check_weather(city: str) -> dict:
    """Look up a city's current weather using Open-Meteo geocoding and weather APIs."""
    query = urllib.parse.urlencode({"name": city, "count": 1, "language": "en"})
    places = json.loads(
        fetch("https://geocoding-api.open-meteo.com/v1/search?" + query)
    )
    if not places.get("results"):
        return {"error": "City not found", "city": city}
    place = places["results"][0]
    query = urllib.parse.urlencode(
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        }
    )
    weather = json.loads(fetch("https://api.open-meteo.com/v1/forecast?" + query))
    return {
        "city": place["name"],
        "country": place.get("country"),
        "timezone": weather["timezone"],
        "utc_offset_seconds": weather["utc_offset_seconds"],
        "current": weather["current"],
        "units": weather["current_units"],
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("../practice_agent/.env"))
    parser.add_argument("--prompt", default="check latest news on AI")
    parser.add_argument("--port", type=int, default=8804)
    parser.add_argument("--ws-port", type=int, default=8805)
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    if not os.getenv("IFM_API_KEY"):
        parser.error("IFM_API_KEY is missing")
    async with AsyncOpenAI(
        api_key=os.environ["IFM_API_KEY"],
        base_url="https://api.ifm.ai/v1",
        timeout=90,
        max_retries=0,
    ) as client:
        graph = build_agent(
            client,
            os.getenv("IFM_MODEL", "IFM/K2-Horizon-375B-A23B"),
            tools=[check_news, check_weather],
        )
        viewer = watch(
            graph,
            port=args.port,
            ws_port=args.ws_port,
            open_browser=not args.no_open_browser,
            theme="dark",
        )
        print(
            f"Viewer: http://localhost:{args.port}/?ws_port={args.ws_port}&theme=dark",
            flush=True,
        )
        try:
            result = await viewer.ainvoke(
                {"messages": [HumanMessage(content=args.prompt)], "steps": 0},
                config={"recursion_limit": 20},
            )
            print("Answer: " + result["messages"][-1].content, flush=True)
        finally:
            if args.trace_file:
                viewer.export_trace(args.trace_file)
        print(
            "Viewer remains open; Resume replays the actual recorded execution.",
            flush=True,
        )
        try:
            await asyncio.Event().wait()
        finally:
            await viewer.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

import requests
import random
from aife_time import now, days_ago, year_start, year_end, iso_date
from aife_utils import retrieve

TAVILY_API_KEYS = [
    retrieve("Tavily"),
    retrieve("Tavily2"),
    retrieve("Tavily3"),
    retrieve("Tavily4")
]

BING_API_KEYS = [
    retrieve("BingKey"),
    retrieve("Bing2Key")
]

SERP_API_KEYS = [
    retrieve("Serp"),
    retrieve("Serp2"),
    retrieve("Serp3"),
    retrieve("Serp4")
]

EXA_API_KEY = retrieve("Exa")


def request_tavily_answer(url, payload):
    for attempt, api_key in enumerate(random.sample(TAVILY_API_KEYS, len(TAVILY_API_KEYS)), 1):
        payload["api_key"] = api_key
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, json=payload, timeout=20).json()
            tavily_answer = response.get("answer")
            tavily_results = response.get("results")
            if tavily_answer or tavily_results:
                return {"tavily_answer": tavily_answer, "tavily_results": tavily_results}
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        print("Failed to get a valid response after maximum retries")
        return {}


def tavily_answer(query):
    url = "https://api.tavily.com/search"
    payload = {
        "query": query,
        "search_depth": "advanced",
        "topic": "general",
        "include_answer": True,
        "include_raw_content": True
    }
    return request_tavily_answer(url, payload)


def request_serp_answer(url, params):
    for attempt, api_key in enumerate(random.sample(SERP_API_KEYS, len(SERP_API_KEYS)), 1):
        params["api_key"] = api_key
        try:
            print(f"Sending request to {url}")
            response = requests.get(url, params=params, timeout=20).json()
            google_answer = response.get("answer_box")
            google_knowledge_graph = response.get("knowledge_graph")
            google_results = response.get("organic_results")
            if google_answer or google_knowledge_graph:
                return {"answer_box": google_answer, "knowledge_graph": google_knowledge_graph}
            elif google_results:
                return {"results": google_results}
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return {}


def google_answer(query):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "location": "County Dublin,Ireland",
        "gl": "ie",
        "safe": "off",
        "no_cache": "true"
    }
    return request_serp_answer(url, params)


def request_tavily(url, payload):
    for attempt, api_key in enumerate(random.sample(TAVILY_API_KEYS, len(TAVILY_API_KEYS)), 1):
        payload["api_key"] = api_key
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, json=payload, timeout=20).json()
            results = response.get("results")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return []


def tavily_by_freshness(query, freshness=None):
    url = "https://api.tavily.com/search"
    payload = {
        "query": query,
        "search_depth": "advanced",
        "topic": "general",
        "time_range": {1: "d", 7: "w", 30: "m"}.get(freshness),
        "include_answer": False,
        "max_results": 70
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("content"),
        "url": item.get("url"),
    } for item in request_tavily(url, payload)]


def tavily_news_by_freshness(query, freshness=None):
    url = "https://api.tavily.com/search"
    payload = {
        "query": query,
        "search_depth": "advanced",
        "topic": "news",
        "days": freshness,
        "include_answer": False,
        "max_results": 100
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("content"),
        "url": item.get("url"),
        "date": iso_date(item.get("published_date"))
    } for item in request_tavily(url, payload)]


def request_bing(url, params):
    for attempt, api_key in enumerate(random.sample(BING_API_KEYS, len(BING_API_KEYS)), 1):
        headers = {
            "Ocp-Apim-Subscription-Key": api_key
        }
        try:
            print(f"Sending request to {url}")
            response = requests.get(url, headers=headers, params=params, timeout=20).json()
            results = response.get("webPages", {}).get("value")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return []


def bing_by_freshness(query, freshness=None):
    url = "https://api.bing.microsoft.com/v7.0/search"
    params = {
        "q": query,
        "responseFilter": "Webpages",
        "freshness": {1: "Day", 7: "Week", 30: "Month"}.get(freshness),
        "mkt": "en-US",
        "safeSearch": "Off",
        "count": 50
    }
    return [{
        "title": item.get("name"),
        "summary": item.get("snippet"),
        "url": item.get("url"),
        "date": iso_date(item.get("datePublished"))
    } for item in request_bing(url, params)]


def bing_by_year_range(query, year_range=None):
    url = "https://api.bing.microsoft.com/v7.0/search"
    params = {
        "q": query,
        "responseFilter": "Webpages",
        "freshness": f"{year_start(year_range[0]).date().isoformat()}..{year_end(year_range[1]).date().isoformat()}" if year_range else None,
        "mkt": "en-US",
        "safeSearch": "Off",
        "count": 50
    }
    return [{
        "title": item.get("name"),
        "summary": item.get("snippet"),
        "url": item.get("url"),
        "date": iso_date(item.get("datePublished"))
    } for item in request_bing(url, params)]


def request_serp(url, params):
    for attempt, api_key in enumerate(random.sample(SERP_API_KEYS, len(SERP_API_KEYS)), 1):
        params["api_key"] = api_key
        try:
            print(f"Sending request to {url}")
            response = requests.get(url, params=params, timeout=20).json()
            results = response.get("organic_results")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return []


def google_by_freshness(query, freshness=None):
    url = "https://serpapi.com/search"
    params = {
        "q": f"{query} after {(days_ago(freshness)).date().isoformat()}" if freshness else query,
        "location": "County Dublin,Ireland",
        "gl": "ie",
        "safe": "off",
        "no_cache": "true",
        "num": 100
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "url": item.get("link"),
        "date": iso_date(item.get("date"))
    } for item in request_serp(url, params)]


def google_by_year_range(query, year_range=None):
    url = "https://serpapi.com/search"
    params = {
        "q": f"{query} after:{year_range[0] - 1} before:{year_range[1] + 1}" if year_range else query,
        "location": "County Dublin,Ireland",
        "gl": "ie",
        "safe": "off",
        "no_cache": "true",
        "num": 100
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "url": item.get("link"),
        "date": iso_date(item.get("date"))
    } for item in request_serp(url, params)]


def google_scholar_by_year_range(query, year_range=None):
    url = "https://serpapi.com/search.json?engine=google_scholar"
    params = {
        "q": query,
        "as_ylo": year_range[0] if year_range else None,
        "as_yhi": year_range[1] if year_range else None,
        "location": "County Dublin,Ireland",
        "gl": "ie",
        "safe": "off",
        "no_cache": "true",
        "num": 20
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "publication_info": item.get("publication_info", {}).get("summary"),
        "url": item.get("link")
    } for item in request_serp(url, params)]


def google_patents_by_year_range(query, year_range=None):
    url = "https://serpapi.com/search.json?engine=google_patents"
    params = {
        "q": query,
        "after": f"publication:{year_start(year_range[0]).year}{year_start(year_range[0]).month:02d}{year_start(year_range[0]).day:02d}" if year_range else None,
        "before": f"publication:{year_end(year_range[1]).year}{year_end(year_range[1]).month:02d}{year_end(year_range[1]).day:02d}" if year_range else None,
        "status": "GRANT",
        "type": "PATENT",
        "no_cache": "true",
        "num": 20
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "url": item.get("pdf"),
        "grant_date": iso_date(item.get("grant_date"))
    } for item in request_serp(url, params)]


def duckduckgo_by_freshness(query, freshness=None):
    url = "https://serpapi.com/search.json?engine=duckduckgo"
    params = {
        "q": query,
        "df": {1: "d", 7: "w", 30: "m"}.get(freshness),
        "safe": -2,
        "no_cache": "true",
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "url": item.get("link"),
        "date": iso_date(item.get("date"))
    } for item in request_serp(url, params)]


def duckduckgo_by_year_range(query, year_range=None):
    url = "https://serpapi.com/search.json?engine=duckduckgo"
    params = {
        "q": query,
        "df": f"{year_start(year_range[0]).date().isoformat()}..{year_end(year_range[1]).date().isoformat()}" if year_range else None,
        "safe": -2,
        "no_cache": "true",
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "url": item.get("link"),
        "date": iso_date(item.get("date"))
    } for item in request_serp(url, params)]


def request_exa(url, headers, payload):
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=payload, timeout=20).json()
            results = response.get("results")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return []


def exa_by_freshness(query, freshness=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{days_ago(freshness).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if freshness else None,
        "endPublishedDate": f"{now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z",
        "numResults": 25
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("summary"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for item in request_exa(url, headers, payload)]


def exa_by_year_range(query, year_range=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{year_start(year_range[0]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if year_range else None,
        "endPublishedDate": f"{year_end(year_range[1]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if year_range else None,
        "numResults": 25
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("summary"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for item in request_exa(url, headers, payload)]


def exa_news_by_freshness(query, freshness=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "category": "news",
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{days_ago(freshness).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if freshness else None,
        "endPublishedDate": f"{now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z",
        "numResults": 25
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("summary"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for item in request_exa(url, headers, payload)]


def exa_paper_by_freshness(query, freshness=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "category": "research paper",
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{days_ago(freshness).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if freshness else None,
        "endPublishedDate": f"{now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z",
        "numResults": 25
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("summary"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for item in request_exa(url, headers, payload)]


def exa_paper_by_year_range(query, year_range=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "category": "research paper",
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{year_start(year_range[0]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if year_range else None,
        "endPublishedDate": f"{year_end(year_range[1]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if year_range else None,
        "numResults": 25
    }
    return [{
        "title": item.get("title"),
        "summary": item.get("summary"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for item in request_exa(url, headers, payload)]

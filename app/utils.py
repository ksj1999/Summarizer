import requests
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

load_dotenv()  # .env 파일의 환경 변수를 로드

# 환경 변수에서 API 키를 가져옴
GPT_API_KEY = os.getenv('GPT4_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY')
NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')
DEEPSEARCH_API_KEY = os.getenv('DEEPSEARCH_API_KEY')  # DeepSearch API 키

SERP_API_URL = 'https://serpapi.com/search.json'
NAVER_API_URL = 'https://openapi.naver.com/v1/search/news.json'
DEEPSEARCH_API_URL = 'https://api.deepsearch.com/v1/compute'

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-summarization")

def summarize_news(news_text):
    inputs = tokenizer(news_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_search_query_with_gpt(summary):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GPT_API_KEY}'
    }
    data = {
        'model': 'gpt-4o',
        'messages': [
            {"role": "system", "content": "You are a helpful assistant that generates search queries based on summaries."},
            {"role": "user", "content": f"Generated Summary: {summary}. Generate a short search query for Google."}
        ],
        'max_tokens': 100
    }
    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data, timeout=10)
        return response.json()
    except (requests.exceptions.JSONDecodeError, requests.exceptions.RequestException) as e:
        return {"error": f"Failed to fetch search query from GPT API: {str(e)}"}

def search_google_scholar(query):
    params = {
        'engine': 'google_scholar',
        'q': query,
        'api_key': SERP_API_KEY,
        'num': 2  # 결과 수를 2개로 제한
    }
    try:
        response = requests.get(SERP_API_URL, params=params, timeout=10)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        return response.json()
    except (requests.exceptions.JSONDecodeError, requests.exceptions.RequestException) as e:
        return {"error": f"Failed to fetch results from Google Scholar: {str(e)}"}

def search_naver_news(query):
    headers = {
        'X-Naver-Client-Id': NAVER_CLIENT_ID,
        'X-Naver-Client-Secret': NAVER_CLIENT_SECRET
    }
    params = {
        'query': query,
        'display': 2  # 결과 수를 2개로 제한
    }
    try:
        response = requests.get(NAVER_API_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        data = response.json()
        if 'items' not in data or len(data['items']) == 0:
            return {"error": "No results found in Naver News."}
        return data
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch results from Naver News: {str(e)}"}

def search_combined(query):
    google_scholar_results = search_google_scholar(query)
    naver_news_results = search_naver_news(query)
    return google_scholar_results, naver_news_results

def search_deepsearch(query):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEEPSEARCH_API_KEY}'
    }
    params = {
        'input': query
    }
    try:
        response = requests.get(DEEPSEARCH_API_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        return response.json()
    except (requests.exceptions.JSONDecodeError, requests.exceptions.RequestException) as e:
        return {"error": f"Failed to fetch results from DeepSearch: {str(e)}"}

def summarize_with_gpt(content):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GPT_API_KEY}'
    }
    data = {
        'model': 'gpt-4o',
        'messages': [
            {"role": "system", "content": "You are a helpful assistant that summarizes articles in Korean."},
            {"role": "user", "content": f"다음 내용을 요약해 주세요: {content}"}
        ],
        'max_tokens': 1000
    }
    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data, timeout=10)
        return response.json()
    except (requests.exceptions.JSONDecodeError, requests.exceptions.RequestException) as e:
        return {"error": f"Failed to fetch summary from GPT API: {str(e)}"}

def create_report(summaries):
    report = "\n\n".join([f"자료 {i+1}:\n{summary}" for i, summary in enumerate(summaries)])
    return report

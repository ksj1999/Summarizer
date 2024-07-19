import requests
import os
from dotenv import load_dotenv
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import PyPDF2
import chromadb
from chromadb.config import Settings

# 로깅 설정
logging.basicConfig(level=logging.ERROR, filename='app_errors.log')

# 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
GPT_API_KEY = os.getenv('GPT4_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY')
NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')

if not all([GPT_API_KEY, SERP_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET]):
    raise ValueError("One or more environment variables are missing.")

SERP_API_URL = 'https://serpapi.com/search.json'
NAVER_API_URL = 'https://openapi.naver.com/v1/search/news.json'

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-summarization")

# ChromaDB 초기화
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db/"
))
collection = chroma_client.get_or_create_collection("text_collection")

def summarize_news(news_text):
    try:
        inputs = tokenizer(news_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logging.error(f"Error in summarize_news: {str(e)}")
        return None

def generate_search_query_with_gpt(summary):
    return gpt_request(
        prompt=f"Generated Summary: {summary}. Generate a short search query for Google.",
        system_message="You are a helpful assistant that generates search queries based on summaries.",
        max_tokens=100
    )

def search_google_scholar(query):
    return search_request(
        url=SERP_API_URL,
        params={
            'engine': 'google_scholar',
            'q': query,
            'api_key': SERP_API_KEY,
            'num': 5
        },
        error_message="Failed to fetch results from Google Scholar"
    )

def search_naver_news(query):
    headers = {
        'X-Naver-Client-Id': NAVER_CLIENT_ID,
        'X-Naver-Client-Secret': NAVER_CLIENT_SECRET
    }
    params = {
        'query': query,
        'display': 5
    }
    return search_request(
        url=NAVER_API_URL,
        headers=headers,
        params=params,
        error_message="Failed to fetch results from Naver News"
    )

def search_google(query):
    return search_request(
        url=SERP_API_URL,
        params={
            'engine': 'google',
            'q': query,
            'api_key': SERP_API_KEY,
            'num': 2
        },
        error_message="Failed to fetch results from Google"
    )

def search_naver(query):
    return search_request(
        url=SERP_API_URL,
        params={
            'engine': 'naver',
            'query': query,
            'api_key': SERP_API_KEY,
            'num': 2
        },
        error_message="Failed to fetch results from Naver"
    )

def search_combined(query):
    google_scholar_results = search_google_scholar(query)
    naver_news_results = search_naver_news(query)
    google_results = search_google(query)
    naver_search_results = search_naver(query)
    return google_scholar_results, naver_news_results, google_results, naver_search_results

def analyze_with_gpt(summary, combined_content):
    return gpt_request(
        prompt=(
            f"다음은 기사 요약입니다: {summary}\n\n"
            f"다음은 검색 결과입니다:\n{combined_content}\n\n"
            "이 정보를 바탕으로 간단한 분석을 제공하고, 정량적 사실을 3가지 리스트업 해주세요."
        ),
        system_message="You are a helpful assistant that provides a brief analysis based on summaries and search results.",
        max_tokens=500
    )

def classify_steep_with_gpt(summary):
    return gpt_request(
        prompt=f"Generated Summary: \"{summary}\". Based on the above news summary, determine whether it falls under Social, Technological, Economic, Environmental, or Political (STEEP). Respond with only the category name in title case (e.g., Economic).",
        system_message="You are a helpful assistant that classifies news summaries into STEEP categories.",
        max_tokens=20
    )

def extract_and_explain_keywords(summary):
    keyword_extraction = gpt_request(
        prompt=f"다음은 기사 요약입니다: \"{summary}\". 위 요약에서 한두 개의 핵심 기술 또는 전문 용어를 추출하고, 각 용어에 대한 간략한 설명을 제공해 주세요.",
        system_message="당신은 요약에서 핵심 기술 용어를 추출하고 설명을 제공하는 유용한 도우미입니다.",
        max_tokens=200
    )
    if 'choices' in keyword_extraction and len(keyword_extraction['choices']) > 0:
        return keyword_extraction['choices'][0]['message']['content']
    else:
        return "핵심 키워드를 추출할 수 없습니다."

def gpt_request(prompt, system_message, max_tokens):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GPT_API_KEY}'
    }
    data = {
        'model': 'gpt-4o',
        'messages': [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        'max_tokens': max_tokens
    }
    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data, timeout=100)
        return response.json()
    except (requests.exceptions.JSONDecodeError, requests.exceptions.RequestException) as e:
        logging.error(f"Error in gpt_request: {str(e)}")
        return {"error": f"Failed to fetch response from GPT API: {str(e)}"}

def search_request(url, params, headers=None, error_message="Failed to fetch search results"):
    try:
        response = requests.get(url, params=params, headers=headers, timeout=100)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.JSONDecodeError, requests.exceptions.RequestException) as e:
        logging.error(f"{error_message}: {str(e)}")
        return {"error": f"{error_message}: {str(e)}"}

def select_related_company(summary):
    return gpt_request(
        prompt=f"Based on the following news summary: \"{summary}\", provide only the stock ticker symbol of the most relevant company. Respond with only the ticker symbol.",
        system_message="You are a helpful assistant that selects the most relevant company's stock ticker symbol based on the news summary.",
        max_tokens=10  # Stock ticker
    )

def process_text_and_store_vectors(text, source_name):
    try:
        document_id = f"doc_{source_name}"
        collection.add(
            documents=[text],
            metadatas=[{"source": source_name}],
            ids=[document_id]
        )
        return document_id
    except Exception as e:
        logging.error(f"Error in process_text_and_store_vectors: {str(e)}")
        raise e

def generate_rag_report(summary):
    try:
        results = collection.query(
            query_texts=[summary],
            n_results=5
        )
        
        combined_text = summary + "\n\n" + "\n\n".join([result['documents'][0] for result in results['documents']])
        
        rag_report_response = gpt_request(
            prompt=f"Based on the following summary and related content, provide a detailed report: {combined_text}",
            system_message="You are a helpful assistant that generates a detailed report based on summary and related documents.",
            max_tokens=1000
        )
        
        if 'choices' in rag_report_response and len(rag_report_response['choices']) > 0:
            return rag_report_response['choices'][0]['message']['content']
        else:
            return "RAG 보고서를 생성할 수 없습니다."
    except Exception as e:
        logging.error(f"Error in generate_rag_report: {str(e)}")
        raise e

def process_pdf_and_store_vectors(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        all_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            all_text += page.extract_text()
        
        document_id = process_text_and_store_vectors(all_text, pdf_file.name)
        return all_text
    except Exception as e:
        logging.error(f"Error in process_pdf_and_store_vectors: {str(e)}")
        raise e

# 핵심 키워드 설명 추가 예제
def generate_analysis_with_keywords(summary, combined_content):
    analysis_response = analyze_with_gpt(summary, combined_content)
    keyword_explanation = extract_and_explain_keywords(summary)
    if "error" in analysis_response:
        return {"error": analysis_response["error"]}
    elif "error" in keyword_explanation:
        return {"error": keyword_explanation["error"]}
    else:
        analysis = analysis_response['choices'][0]['message']['content']
        return {
            "analysis": analysis,
            "keywords": keyword_explanation
        }

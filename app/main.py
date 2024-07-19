import streamlit as st
import yfinance as yf
import pandas as pd
import logging
from utils import summarize_news, generate_search_query_with_gpt, search_combined, analyze_with_gpt, classify_steep_with_gpt, select_related_company, process_pdf_and_store_vectors, process_text_and_store_vectors, generate_rag_report, generate_analysis_with_keywords

import logging
from utils import process_pdf_and_store_vectors, process_text_and_store_vectors

logging.basicConfig(level=logging.DEBUG, filename='app_debug.log', format='%(asctime)s %(levelname)s:%(message)s')

st.image("assets/full_logo_cut.png", width=200)  # 로고 이미지 경로

# PDF 업로드 섹션
st.sidebar.markdown("<h2 style='color:#0E1B4A;'>PDF 업로드</h2>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("PDF 파일을 업로드", type="pdf")

pdf_text = None
if uploaded_file:
    try:
        logging.debug("PDF 파일 업로드됨: %s", uploaded_file.name)
        pdf_text = process_pdf_and_store_vectors(uploaded_file)
        st.sidebar.success("PDF 파일이 성공적으로 업로드 및 벡터화되었습니다.")
    except Exception as e:
        logging.error("PDF 파일 처리 중 오류 발생: %s", str(e))
        st.sidebar.error(f"PDF 파일 처리 중 오류 발생: {str(e)}")

st.markdown("<h2 style='color:#0E1B4A;'>원문</h2>", unsafe_allow_html=True)

# 사용자 입력
news_text = st.text_area("원문", height=400, label_visibility="hidden")

def display_error(message):
    st.error(message)

def display_warning(message):
    st.warning(message)

def format_results(results):
    formatted_results = ""
    for i, result in enumerate(results):
        formatted_results += f"{i+1}. {result['title']} ({result['link']})<br><br>"
    return formatted_results

def get_first_four_sentences(text):
    if text:
        sentences = text.split('.')
        return '.'.join(sentences[:4]) + '.'
    return text

if st.button("리서치 자료 생성"):
    with st.spinner('처리 중...'):
        summary = summarize_news(news_text)
        if not summary:
            display_error("요약 생성에 실패했습니다.")
        else:
            steep_classification_response = classify_steep_with_gpt(summary)
            if "error" in steep_classification_response:
                display_error(steep_classification_response["error"])
            else:
                steep_classification = steep_classification_response['choices'][0]['message']['content'].strip().lower()
                
                search_query_response = generate_search_query_with_gpt(summary)
                if "error" in search_query_response:
                    display_error(search_query_response["error"])
                else:
                    search_query = search_query_response['choices'][0]['message']['content']

                    google_scholar_results, naver_news_results, google_results, naver_search_results = search_combined(search_query)

                    if "error" in google_scholar_results:
                        display_error(google_scholar_results["error"])
                    elif "error" in naver_news_results:
                        display_error(naver_news_results["error"])
                    else:
                        google_scholar_content = format_results(google_scholar_results.get('organic_results', [])[:3])
                        naver_news_content = format_results(naver_news_results.get('items', [])[:3])

                        google_scholar_content = google_scholar_content or "구글 스칼라에서 결과를 찾을 수 없습니다."
                        naver_news_content = naver_news_content or "네이버 뉴스에서 결과를 찾을 수 없습니다."

                        combined_content = f"구글 스칼라:\n{google_scholar_content}\n\n네이버 뉴스:\n{naver_news_content}"
                        combined_content += f"\n\n구글 검색 결과:\n{format_results(google_results.get('organic_results', [])[:3])}"
                        combined_content += f"\n\n네이버 검색 결과:\n{format_results(naver_search_results.get('items', [])[:3])}"
                        combined_content = combined_content[:4000]

                        analysis_with_keywords = generate_analysis_with_keywords(summary, combined_content)
                        
                        if "error" in analysis_with_keywords:
                            display_error(f"오류: {analysis_with_keywords['error']}")
                        else:
                            analysis = analysis_with_keywords['analysis']
                            keywords = analysis_with_keywords['keywords']

                            st.markdown("<h2 style='color:#0E1B4A;'>요약</h2>", unsafe_allow_html=True)
                            st.text_area("요약내용", summary, height=110, label_visibility="hidden")

                            # 검색 결과 표시
                            st.markdown("<h2 style='color:#0E1B4A;'>리서치 자료</h2>", unsafe_allow_html=True)

                            st.markdown(f"""
                            <div class='results-container'>
                                <div class='overview'>
                                    <h4>개요</h4>
                                    <p>{analysis}</p>
                                </div>
                                <div class='keywords'>
                                    <h4>핵심 키워드 설명</h4>
                                    <p>{keywords}</p>
                                </div>
                                <div class='google-scholar'>
                                    <h4>학술 자료</h4>
                                    <p>{google_scholar_content}</p>
                                </div>
                                <div class='naver-news'>
                                    <h4>관련 자료</h4>
                                    <p>{naver_news_content}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if steep_classification == 'economic':
                                company_response = select_related_company(summary)
                                if "error" in company_response:
                                    display_error(company_response["error"])
                                else:
                                    ticker = company_response['choices'][0]['message']['content'].strip()

                                    try:
                                        # yfinance를 통해 주식 티커에 대한 데이터 가져오기
                                        ticker_data = yf.Ticker(ticker)
                                        ticker_info = ticker_data.info

                                        name = ticker_info.get('longName', 'N/A')
                                        summary = get_first_four_sentences(ticker_info.get('longBusinessSummary', 'N/A'))
                                        fullTimeEmployees = ticker_info.get('fullTimeEmployees', 'N/A')
                                        industry = ticker_info.get('industry', 'N/A')
                                        sector = ticker_info.get('sector', 'N/A')
                                        currentPrice = ticker_info.get('currentPrice', 'N/A')
                                        openPrice = ticker_info.get('open', 'N/A')
                                        fiftytwoLow = ticker_info.get('fiftyTwoWeekLow', 'N/A')
                                        fiftytwoHigh = ticker_info.get('fiftyTwoWeekHigh', 'N/A')
                                        recommKey = ticker_info.get('recommendationKey', 'N/A')
                                        meanTarget = ticker_info.get('targetMeanPrice', 'N/A')
                                        lowTarget = ticker_info.get('targetLowPrice', 'N/A')
                                        highTarget = ticker_info.get('targetHighPrice', 'N/A')
                                        debtToEquity = ticker_info.get('debtToEquity', 'N/A')
                                        totalDebt = ticker_info.get('totalDebt', 'N/A')
                                        totalAsset = ticker_info.get('totalAssets', 'N/A')
                                        marketCap = ticker_info.get('marketCap', 'N/A')
                                        bookValue = ticker_info.get('bookValue', 'N/A')
                                        totalRevenue = ticker_info.get('totalRevenue', 'N/A')
                                        totalCash = ticker_info.get('totalCash', 'N/A')
                                        operatingMargins = ticker_info.get('operatingMargins', 'N/A')
                                        profitMargins = ticker_info.get('profitMargins', 'N/A')
                                        revenueGrowth = ticker_info.get('revenueGrowth', 'N/A')
                                        roe = ticker_info.get('returnOnEquity', 'N/A')
                                        currentRatio = ticker_info.get('currentRatio', 'N/A')
                                        quickRatio = ticker_info.get('quickRatio', 'N/A')
                                        recommendationMean = ticker_info.get('recommendationMean', 'N/A')
                                        sharesShort = ticker_info.get('sharesShort', 'N/A')
                                        priceToBook = ticker_info.get('priceToBook', 'N/A')
                                        trailingEps = ticker_info.get('trailingEps', 'N/A')
                                        revPerEmployee = round(totalRevenue / fullTimeEmployees, 2) if fullTimeEmployees and totalRevenue else 'N/A'

                                        recommendations = ticker_data.recommendations

                                        st.markdown("<h2 style='color:#0E1B4A;'>재무 분석 보고서</h2>", unsafe_allow_html=True)
                                        with st.container():
                                            st.markdown(f"""
                                            <div class='results-container'>
                                                <h3>회사: {name}</h3>
                                                <h4>비즈니스 요약</h4>
                                                <p style='text-align:justify;'>{summary}</p>
                                                <br>
                                                <div class='company-metadata'>
                                                    <h4>회사 메타 데이터</h4>
                                                    {pd.DataFrame({
                                                        '시가총액': [f'${marketCap:,}' if marketCap != 'N/A' else 'N/A'],
                                                        '섹터': [sector],
                                                        '산업': [industry],
                                                        '전체 직원 수': [f'{fullTimeEmployees:,}' if fullTimeEmployees != 'N/A' else 'N/A'],
                                                        '직원당 매출': [f'${revPerEmployee:,}' if revPerEmployee != 'N/A' else 'N/A']
                                                    }).to_html(index=False, classes='dataframe')}
                                                </div>
                                                <hr><br>
                                                <div class='financial-info'>
                                                    <div style='display: flex; justify-content: space-between;'>
                                                        <div style='width: 48%;'>
                                                            <h4><strong>가격 정보</strong></h4>
                                                            <p><strong>현재 가격:</strong>  {'${:,.2f}'.format(currentPrice) if currentPrice != 'N/A' else 'N/A'}</p>
                                                            <p><strong>52주 최저가:</strong>  {'${:,.2f}'.format(fiftytwoLow) if fiftytwoLow != 'N/A' else 'N/A'}</p>
                                                            <p><strong>52주 최고가:</strong>  {'${:,.2f}'.format(fiftytwoHigh) if fiftytwoHigh != 'N/A' else 'N/A'}</p>
                                                        </div>
                                                        <div style='width: 48%;'>
                                                            <h4><strong>목표 가격</strong></h4>
                                                            <p><strong>목표 평균 가격:</strong> {'${:,.2f}'.format(meanTarget) if meanTarget != 'N/A' else 'N/A'}</p>
                                                            <p><strong>목표 최저 가격:</strong> {'${:,.2f}'.format(lowTarget) if lowTarget != 'N/A' else 'N/A'}</p>
                                                            <p><strong>목표 최고 가격:</strong> {'${:,.2f}'.format(highTarget) if highTarget != 'N/A' else 'N/A'}</p>
                                                        </div>
                                                    </div>
                                                    <h4><strong>추천:</strong> <strong style='color: #fafafa; border-radius: 4px; padding:2px 10px; background: skyblue; font-weight: 800;'>{recommKey}</strong></h4>
                                                </div>
                                                <hr><br>
                                                <div class='financial-health'>
                                                    <h4><strong>재무 상태</strong></h4>
                                                    <div style='display: flex; justify-content: space-between;'>
                                                        <div style='width: 48%;'>
                                                            <p><strong>총 자산:</strong>  {'${:,.2f}'.format(totalAsset) if totalAsset != 'N/A' else 'N/A'}</p>
                                                            <p><strong>총 부채:</strong>  {'${:,.2f}'.format(totalDebt) if totalDebt != 'N/A' else 'N/A'}</p>
                                                            <p><strong>총 현금:</strong>  {'${:,.2f}'.format(totalCash) if totalCash != 'N/A' else 'N/A'}</p>
                                                            <p><strong>장부가치:</strong>  {'${:,.2f}'.format(bookValue) if bookValue != 'N/A' else 'N/A'}</p>
                                                            <p><strong>총 매출:</strong>  {'${:,.2f}'.format(totalRevenue) if totalRevenue != 'N/A' else 'N/A'}</p>
                                                            <p><strong>매출 성장률:</strong> {revenueGrowth}</p>
                                                            <p><strong>영업 이익률:</strong> {operatingMargins}</p>
                                                            <p><strong>순이익률:</strong> {profitMargins}</p>
                                                        </div>
                                                        <div style='width: 48%;'>
                                                            <p><strong>자기자본이익률:</strong> {roe}</p>
                                                            <p><strong>현재 비율:</strong> {currentRatio}</p>
                                                            <p><strong>당좌 비율:</strong> {quickRatio}</p>
                                                            <p><strong>추천 평균:</strong> {recommendationMean}</p>
                                                            <p><strong>공매도 주식:</strong> {sharesShort}</p>
                                                            <p><strong>주가 대비 장부가:</strong> {priceToBook}</p>
                                                            <p><strong>주당순이익:</strong> {trailingEps}</p>
                                                        </div>
                                                    </div>
                                                </div>
                                                <hr><br>
                                                <div class='recommendations'>
                                                    <h4><strong>총 추천 등급</strong></h4>
                                                    {recommendations['To Grade'].value_counts().to_frame().to_html(index=True, classes='dataframe') if 'To Grade' in recommendations.columns else '추천 등급 정보를 사용할 수 없습니다.'}
                                                </div>
                                                <hr><br>
                                                <div class='recent-news'>
                                                    <h4><strong>최근 뉴스</strong></h4>
                                                    {"<br>".join([f"{article['title']} (<a href='{article['link']}'>{article['link']}</a>)" for article in ticker_data.news[:3]])}
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                                        # 차트 데이터 표시
                                        st.markdown("##### 종가 차트")
                                        ticker_df = ticker_data.history(period='max')
                                        st.line_chart(ticker_df['Close'])

                                        st.markdown("##### 거래량 차트")
                                        st.bar_chart(ticker_df['Volume'])

                                        st.markdown("##### 재무제표")
                                        income_statement = ticker_data.financials
                                        st.write(income_statement)

                                        st.markdown("##### 대차대조표")
                                        balance_sheet = ticker_data.balance_sheet
                                        st.write(balance_sheet)

                                        st.markdown("##### 현금흐름표")
                                        cashflow = ticker_data.cashflow
                                        st.write(cashflow)

                                    except Exception as e:
                                        display_error(f"{ticker}에 대한 재무 데이터를 가져오는 데 실패했습니다: {str(e)}")
                            else:
                                # PDF 파일 업로드 및 RAG 보고서 생성
                                if uploaded_file and pdf_text:
                                    try:
                                        with st.spinner('PDF 파일 처리 중...'):
                                            rag_report = generate_rag_report(summary, pdf_text)
                                            st.markdown("<h2 style='color:#0E1B4A;'>RAG 보고서</h2>", unsafe_allow_html=True)
                                            st.markdown(rag_report, unsafe_allow_html=True)
                                    except Exception as e:
                                        display_error(f"PDF 파일 처리 중 오류 발생: {str(e)}")

# 스타일 추가
st.markdown("""
<style>
    .stTextArea textarea {
        border: 3px solid #0E1B4A;
        background-color: #f0f2f6;
    }
    h2 {
        color: #0E1B4A;
    }
    .results-container {
        background-color: #f0f2f6;
        padding: 15px;
        border: 3px solid #0E1B4A;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .results-container h3, .results-container h4 {
        color: #000000;
        font-weight: bold;
    }
    .results-container p {
        margin: 0;
    }
    .results-container ul {
        margin: 0;
        padding-left: 20px;
    }
    a {
        text-decoration: none;
        color: #0E1B4A;
    }
    a:hover {
        text-decoration: underline;
    }
    .dataframe {
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

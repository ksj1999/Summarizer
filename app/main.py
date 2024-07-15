import streamlit as st
from utils import summarize_news, generate_search_query_with_gpt, search_combined, search_deepsearch, summarize_with_gpt, create_report

# Streamlit 앱 설정
st.title("Research Paper Summarizer")

# 사용자 입력
news_text = st.text_area("뉴스 원문을 입력하세요:")

if st.button("Summarize and Search"):
    with st.spinner('Generating summary...'):
        summary = summarize_news(news_text)
        if not summary:
            st.error("Failed to generate summary.")
        else:
            st.success(f"Generated Summary: {summary}")
            
            with st.spinner('Generating search query...'):
                search_query_response = generate_search_query_with_gpt(summary)
                if "error" in search_query_response:
                    st.error(search_query_response["error"])
                else:
                    search_query = search_query_response['choices'][0]['message']['content']
                    st.success(f"Generated Search Query: {search_query}")

                    with st.spinner('Searching...'):
                        google_scholar_results, naver_news_results = search_combined(search_query)
                        deepsearch_results = search_deepsearch(search_query)

                        # API 호출 실패 시 오류 메시지 출력
                        if "error" in google_scholar_results:
                            st.error(google_scholar_results["error"])
                        elif "error" in naver_news_results:
                            st.error(naver_news_results["error"])
                        elif "error" in deepsearch_results:
                            st.error(deepsearch_results["error"])
                        else:
                            # 요약을 위한 내용 생성
                            google_scholar_content = "\n".join(
                                [f"{result['snippet']}" for result in google_scholar_results.get('organic_results', [])[:2]]
                            )
                            naver_news_content = "\n".join(
                                [f"{item['description']}" for item in naver_news_results.get('items', [])[:2]]
                            )
                            deepsearch_content = "\n".join(
                                [f"{item['summary']}" for item in deepsearch_results.get('results', [])[:2]]
                            )

                            if not google_scholar_content:
                                google_scholar_content = "No results found from Google Scholar."
                            if not naver_news_content:
                                naver_news_content = "No results found from Naver News."
                            if not deepsearch_content:
                                deepsearch_content = "No results found from DeepSearch."

                            combined_content = f"구글 스칼라:\n{google_scholar_content}\n\n네이버 뉴스:\n{naver_news_content}\n\nDeepSearch:\n{deepsearch_content}"

                            if len(combined_content) > 4000:
                                combined_content = combined_content[:4000]

                            gpt_response = summarize_with_gpt(combined_content)
                            
                            if "error" in gpt_response:
                                st.error(f"Error: {gpt_response['error']}")
                            else:
                                if 'choices' in gpt_response and len(gpt_response['choices']) > 0:
                                    research_materials = gpt_response['choices'][0]['message']['content']
                                    report = create_report([research_materials])
                                    st.markdown("## Summary")
                                    st.markdown(f"<div class='summary'>{report}</div>", unsafe_allow_html=True)
                                else:
                                    st.warning("No research materials found.")

                            # 검색 결과 표시
                            st.markdown("## Search Results")

                            # Google Scholar Results
                            st.markdown("<h2 style='color: #00796b;'>Google Scholar Results</h2>", unsafe_allow_html=True)
                            if google_scholar_content != "No results found from Google Scholar.":
                                for result in google_scholar_results.get('organic_results', [])[:2]:
                                    st.markdown(f"""
                                    <div class='card'>
                                        <h4><a href='{result['link']}' target='_blank'>{result['title']}</a></h4>
                                        <p>{result['snippet']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='card'>{google_scholar_content}</div>", unsafe_allow_html=True)

                            # Naver News Results
                            st.markdown("<h2 style='color: #ff8f00;'>Naver News Results</h2>", unsafe_allow_html=True)
                            if naver_news_content != "No results found from Naver News.":
                                for item in naver_news_results.get('items', [])[:2]:
                                    st.markdown(f"""
                                    <div class='card'>
                                        <h4><a href='{item['link']}' target='_blank'>{item['title']}</a></h4>
                                        <p>{item['description']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='card'>{naver_news_content}</div>", unsafe_allow_html=True)
                            
                            # DeepSearch Results
                            st.markdown("<h2 style='color: #1f4e79;'>DeepSearch Results</h2>", unsafe_allow_html=True)
                            if deepsearch_content != "No results found from DeepSearch.":
                                for item in deepsearch_results.get('results', [])[:2]:
                                    st.markdown(f"""
                                    <div class='card'>
                                        <h4><a href='{item['link']}' target='_blank'>{item['title']}</a></h4>
                                        <p>{item['summary']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='card'>{deepsearch_content}</div>", unsafe_allow_html=True)

# 스타일 추가
st.markdown("""
<style>
    .summary {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .card {
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    .card h4 {
        margin: 0;
        padding: 0.
    }
    .card p {
        margin: 10px 0 0;
        padding: 0.
    }
    a {
        text-decoration: none;
        color: inherit.
    }
    a:hover {
        text-decoration: underline.
    }
</style>
""", unsafe_allow_html=True)

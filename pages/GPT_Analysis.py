# pages/GPT_Analysis.py
import openai
import pandas as pd
import streamlit as st

st.title("GPT 数据分析助手")

# API 密钥输入
openai_api_key = st.text_input("输入你的 OpenAI API 密钥", type="password")

# 数据上传
uploaded_file = st.file_uploader("上传用于分析的数据文件", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("不支持的文件格式")
            df = None

        if df is not None:
            st.subheader("数据预览")
            st.dataframe(df.head())

            if st.button("生成分析思路"):
                if openai_api_key:
                    try:
                        openai.api_key = openai_api_key
                        with st.spinner("GPT 正在分析数据..."):
                            prompt = (
                                    "请根据以下数据集提供详细的数据分析思路。"
                                    "数据集预览如下：\n\n" +
                                    df.head().to_string() +
                                    "\n\n请提供一个全面的数据分析计划，包括可能的分析方法、可视化建议和潜在的洞见。"
                            )
                            response = openai.ChatCompletion.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "你是一个数据分析专家。"},
                                    {"role": "user", "content": prompt}
                                ],
                                max_tokens=10000,
                                n=1,
                                stop=None,
                                temperature=0.7,
                            )
                            analysis = response.choices[0].message['content']
                            st.subheader("分析思路")
                            st.write(analysis)
                    except Exception as api_error:
                        st.error(f"调用OpenAI API时出错: {api_error}")
                else:
                    st.warning("请先输入你的 OpenAI API 密钥。")

    except Exception as e:
        st.error(f"无法读取文件: {e}")
else:
    st.info("请上传一个数据文件进行分析。")

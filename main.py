import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
import pdfkit

# 设置页面配置
st.set_page_config(page_title="数据分析Web应用", layout="wide")
st.title("数据分析工具")

# 上传数据
st.sidebar.header("上传数据")
uploaded_file = st.sidebar.file_uploader("选择一个数据文件", type=["csv", "xlsx", "json"])

def load_data(uploaded_file):
    """加载上传的数据文件"""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        return pd.read_json(uploaded_file)
    else:
        st.error("不支持的文件格式")
        return None

def export_to_pdf(html_content):
    """将HTML内容导出为PDF"""
    pdf_file_path = '/mnt/data/exported_report.pdf'
    pdfkit.from_string(html_content, pdf_file_path)
    return pdf_file_path

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        if df is not None:
            st.subheader("数据预览")
            st.dataframe(df.head())

            # 创建一个操作单元
            def operation_cell(df, cell_id=0):
                st.write(f"操作单元格 {cell_id + 1}")
                operation = st.selectbox("选择操作", ["数据清洗", "可视化", "K-Means 聚类"], key=f"operation_{cell_id}")

                # 数据清洗单元
                if operation == "数据清洗":
                    clean_option = st.selectbox("清洗选项", ["删除缺失值行", "填充缺失值", "删除重复行"], key=f"clean_{cell_id}")
                    if clean_option == "删除缺失值行" and st.button("执行操作", key=f"drop_na_{cell_id}"):
                        df = df.dropna()
                        st.success("已删除所有包含缺失值的行。")
                        st.dataframe(df.head())
                    elif clean_option == "填充缺失值":
                        fill_column = st.selectbox("选择要填充缺失值的列", options=df.columns, key=f"fill_column_{cell_id}")
                        fill_value = st.text_input("填充值", "", key=f"fill_value_{cell_id}")
                        if st.button("执行操作", key=f"fill_na_{cell_id}") and fill_value:
                            df[fill_column] = df[fill_column].fillna(fill_value)
                            st.success(f"已将列 `{fill_column}` 的缺失值填充为 `{fill_value}`。")
                            st.dataframe(df.head())
                    elif clean_option == "删除重复行" and st.button("执行操作", key=f"drop_duplicates_{cell_id}"):
                        df = df.drop_duplicates()
                        st.success("已删除重复行。")
                        st.dataframe(df.head())

                # 可视化单元
                elif operation == "可视化":
                    st.session_state['plot_type'] = st.selectbox("选择图表类型", ["散点图", "折线图", "饼图", "箱线图"],
                                                                 index=["散点图", "折线图", "饼图", "箱线图"].index(
                                                                     st.session_state['plot_type']))

                    # 动态生成图表选择器，保留之前的选择
                    if st.session_state['plot_type'] in ["散点图", "折线图"]:
                        st.session_state['x_axis'] = st.selectbox("选择X轴", df.columns,
                                                                  index=df.columns.get_loc(st.session_state['x_axis']))
                        st.session_state['y_axis'] = st.selectbox("选择Y轴", df.columns,
                                                                  index=df.columns.get_loc(st.session_state['y_axis']))
                        st.session_state['hue'] = st.selectbox("颜色分类（可选）", [None] + list(df.columns),
                                                               index=[None] + list(df.columns).index(
                                                                   st.session_state['hue']) if st.session_state[
                                                                   'hue'] else 0)

                    elif st.session_state['plot_type'] == "饼图":
                        st.session_state['hue'] = st.selectbox("选择用于饼图的列", df.columns,
                                                               index=df.columns.get_loc(st.session_state['hue']) if
                                                               st.session_state['hue'] else 0)

                    elif st.session_state['plot_type'] == "箱线图":
                        numeric_columns = df.select_dtypes(include=['float', 'int']).columns
                        st.session_state['y_axis'] = st.selectbox("选择数值列", numeric_columns,
                                                                  index=numeric_columns.get_loc(
                                                                      st.session_state['y_axis']) if st.session_state[
                                                                      'y_axis'] else 0)
                        st.session_state['hue'] = st.selectbox("选择分类列（可选）", [None] + list(df.columns),
                                                               index=[None] + list(df.columns).index(
                                                                   st.session_state['hue']) if st.session_state[
                                                                   'hue'] else 0)

                    # 根据选择的图表类型生成图表
                    if st.button("生成图表"):
                        if st.session_state['plot_type'] == "散点图":
                            fig = px.scatter(df, x=st.session_state['x_axis'], y=st.session_state['y_axis'],
                                             color=st.session_state['hue'])
                        elif st.session_state['plot_type'] == "折线图":
                            fig = px.line(df, x=st.session_state['x_axis'], y=st.session_state['y_axis'],
                                          color=st.session_state['hue'])
                        elif st.session_state['plot_type'] == "饼图":
                            fig = px.pie(df, names=st.session_state['hue'], title=f"{st.session_state['hue']} 分布")
                        elif st.session_state['plot_type'] == "箱线图":
                            fig = px.box(df, x=st.session_state['hue'], y=st.session_state['y_axis'])

                        st.plotly_chart(fig, use_container_width=True)

                # K-Means 聚类单元
                elif operation == "K-Means 聚类":
                    st.subheader("K-Means 聚类")
                    numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
                    selected_columns = st.multiselect("选择用于聚类的列", numeric_columns, key=f"cluster_columns_{cell_id}")
                    n_clusters = st.number_input("选择簇数", min_value=2, max_value=10, value=3, step=1, key=f"clusters_{cell_id}")
                    distance_metric = st.selectbox("选择距离度量", ("欧式距离", "曼哈顿距离"), key=f"distance_metric_{cell_id}")

                    def calculate_distance(data, metric):
                        return data if metric == "欧式距离" else np.abs(data - data.mean())

                    if st.button("显示肘部图", key=f"elbow_{cell_id}") and selected_columns:
                        sse = []
                        transformed_data = calculate_distance(df[selected_columns], distance_metric)
                        for k in range(1, 11):
                            kmeans = KMeans(n_clusters=k, random_state=0)
                            kmeans.fit(transformed_data)
                            sse.append(kmeans.inertia_)
                        fig = px.line(x=range(1, 11), y=sse, labels={'x': '簇数', 'y': '簇内误差平方和 (SSE)'})
                        fig.update_traces(mode='lines+markers')
                        st.plotly_chart(fig)

                    if st.button("执行聚类", key=f"cluster_{cell_id}") and selected_columns:
                        try:
                            transformed_data = calculate_distance(df[selected_columns], distance_metric)
                            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                            df['Cluster'] = kmeans.fit_predict(transformed_data)
                            st.write("聚类结果：")
                            st.dataframe(df[['Cluster'] + selected_columns].head())
                        except Exception as e:
                            st.error(f"聚类时出错: {e}")

                # 添加新操作选项
                add_new_cell = st.button("添加新操作单元", key=f"add_new_cell_{cell_id}")
                if add_new_cell:
                    st.session_state[f"cell_{cell_id + 1}"] = True

                return df

            # 初始化第一个单元格
            if "cell_0" not in st.session_state:
                st.session_state["cell_0"] = True

            # 渲染所有操作单元格
            current_df = df
            cell_id = 0
            while f"cell_{cell_id}" in st.session_state:
                current_df = operation_cell(current_df, cell_id)
                cell_id += 1

            # 导出PDF
            st.subheader("导出分析报告")
            if st.button("生成PDF"):
                html_content = st.get_report_ctx().report_html
                pdf_file_path = export_to_pdf(html_content)
                with open(pdf_file_path, "rb") as pdf_file:
                    st.download_button(label="下载 PDF 报告", data=pdf_file, file_name="数据分析报告.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"无法读取文件: {e}")
else:
    st.info("请通过左侧边栏上传一个数据文件")

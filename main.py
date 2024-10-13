# main.py
import pandas as pd
import plotly.express as px
import streamlit as st

# 设置页面配置
st.set_page_config(page_title="数据分析Web应用", layout="wide")

st.title("数据分析工具")

st.sidebar.header("上传数据")

uploaded_file = st.sidebar.file_uploader("选择一个数据文件", type=["csv", "xlsx", "json"])

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

            st.subheader("数据清洗")

            # 显示原始数据的缺失值情况
            st.write("缺失值情况：")
            st.write(df.isnull().sum())

            # 删除缺失值
            if st.button("删除所有包含缺失值的行"):
                df = df.dropna()
                st.success("已删除所有包含缺失值的行。")
                st.dataframe(df.head())

            # 填充缺失值
            st.markdown("### 填充缺失值")
            fill_column = st.selectbox("选择要填充缺失值的列", options=df.columns)
            fill_value = st.text_input("填充值（如：0，'未知'）", "")
            if st.button("填充缺失值"):
                if fill_value:
                    df[fill_column] = df[fill_column].fillna(fill_value)
                    st.success(f"已将列 `{fill_column}` 的缺失值填充为 `{fill_value}`。")
                    st.dataframe(df.head())
                else:
                    st.warning("请提供一个填充值。")

            # 删除重复数据
            if st.button("删除重复行"):
                df = df.drop_duplicates()
                st.success("已删除重复行。")
                st.dataframe(df.head())

            st.subheader("数据可视化")

            plot_type = st.selectbox("选择图表类型", ["散点图", "折线图", "饼图", "箱线图"])

            if plot_type == "散点图":
                st.markdown("### 散点图")
                x_axis = st.selectbox("选择X轴", df.columns)
                y_axis = st.selectbox("选择Y轴", df.columns)
                hue = st.selectbox("颜色分类（可选）", [None] + list(df.columns))

                if st.button("生成散点图"):
                    if hue:
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=hue)
                    else:
                        fig = px.scatter(df, x=x_axis, y=y_axis)
                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "折线图":
                st.markdown("### 折线图")
                x_axis = st.selectbox("选择X轴", df.columns, key="line_x")
                y_axis = st.selectbox("选择Y轴", df.columns, key="line_y")
                hue = st.selectbox("颜色分类（可选）", [None] + list(df.columns), key="line_hue")

                if st.button("生成折线图"):
                    if hue:
                        fig = px.line(df, x=x_axis, y=y_axis, color=hue)
                    else:
                        fig = px.line(df, x=x_axis, y=y_axis)
                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "饼图":
                st.markdown("### 饼图")
                pie_column = st.selectbox("选择用于饼图的列", df.columns)
                if st.button("生成饼图"):
                    fig = px.pie(df, names=pie_column, title=f"{pie_column} 分布")
                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "箱线图":
                st.markdown("### 箱线图")
                box_column = st.selectbox("选择数值列", df.select_dtypes(include=['float', 'int']).columns)
                box_category = st.selectbox("选择分类列（可选）", [None] + list(df.columns))

                if st.button("生成箱线图"):
                    if box_category:
                        fig = px.box(df, x=box_category, y=box_column)
                    else:
                        fig = px.box(df, y=box_column)
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("描述性统计")
            if st.checkbox("显示描述性统计"):
                st.write(df.describe())

    except Exception as e:
        st.error(f"无法读取文件: {e}")
else:
    st.info("请通过左侧边栏上传一个数据文件。")

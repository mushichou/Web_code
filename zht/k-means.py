from sklearn.cluster import KMeans
import streamlit as st
st.subheader("K-Means 聚类")

# 获取数值型的列供用户选择
numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
selected_columns = st.multiselect("选择用于聚类的列", numeric_columns)
n_clusters = st.number_input("选择簇数", min_value=2, max_value=10, value=3, step=1)

# 距离度量选择
distance_metric = st.selectbox("选择距离度量", ("欧式距离", "曼哈顿距离"))

# 计算指定距离
def calculate_distance(data, metric):
    if metric == "欧式距离":
        return data
    elif metric == "曼哈顿距离":
        return np.abs(data - data.mean())

if st.button("显示肘部图") and selected_columns:
    # 计算不同簇数下的SSE
    sse = []
    transformed_data = calculate_distance(df[selected_columns], distance_metric)
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(transformed_data)
        sse.append(kmeans.inertia_)
    
    # 绘制肘部图
    st.write("肘部图")
    fig = px.line(x=range(1, 11), y=sse, labels={'x': '簇数', 'y': '簇内误差平方和 (SSE)'})
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig)

if st.button("执行聚类") and selected_columns:
    try:
        transformed_data = calculate_distance(df[selected_columns], distance_metric)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        # 聚类
        df['Cluster'] = kmeans.fit_predict(transformed_data)

        st.write("聚类结果：")
        st.dataframe(df[['Cluster'] + selected_columns].head())

        # 每个簇的描述性统计
        st.write("各个簇的描述统计：")
        cluster_description = df.groupby('Cluster')[selected_columns].describe()
        st.dataframe(cluster_description)
    except Exception as e:
        st.error(f"聚类时出错: {e}")
else:
    st.info("请先选择用于聚类的列和簇数")
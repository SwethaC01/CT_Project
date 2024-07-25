import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Industrial Resources",
    layout="wide",
    initial_sidebar_state="expanded")

with st.sidebar:
    selected = option_menu(None,
                        ["Home","Data Visualization","Geo Visualization"],
                        icons=["house-door-fill","bar-chart","map"],
                        default_index=0,
                        orientation="vertical",
                        styles={"nav-link":{"font-size": "20px", "text-align": "center", "margin": "0px", "--hover-color": "#00008B"},
                        "icon": {"font-size": "20px"},
                        "container" : {"max-width": "6000px"},
                        "nav-link-selected": {"background-color": "#00008B"}})

scrolling_text = "<h1 style='color:Blue; font-style:bold ; font-weight: bold;'><marquee>INDUSTRIAL HUMAN RESOURCE GEO-VISUALIZATION</marquee></h1>"
st.markdown(scrolling_text, unsafe_allow_html=True)

directory_path = "D:\Swetha Documents\RE-CT_PROJECT\DataSets"

csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

dataframes = [pd.read_csv(os.path.join(directory_path, file), encoding='latin1') for file in csv_files]

merged_df = pd.concat(dataframes, ignore_index=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------#
# Separate state and district names
merged_df[['STATE', 'District']] = merged_df['India/States'].str.split(' - ', expand=True)

# Function to separate state and district names
def separate_state_district(row):
    # Split the string based on the separator '-'
    parts = row.split(' - ')

    # If the first part is in uppercase (assumed to be state name), return it
    if parts[0].isupper():
        return parts[0]
    else:
        return None

# Apply the function to create a new column for state names
merged_df['State Name'] = merged_df['District'].apply(separate_state_district)

# Filter out None values and then print unique state names with commas
state_names = merged_df['State Name'].dropna().unique()
print(", ".join(state_names))

# Create a mapping dictionary for state names
state_name_mapping = {
    'ANDHRA PRADESH': 'Andhra Pradesh',
    'ARUNACHAL PRADESH': 'Arunachal Pradesh',
    'ASSAM': 'Assam',
    'BIHAR': 'Bihar',
    'CHHATTISGARH': 'Chhattisgarh',
    'GOA': 'Goa',
    'GUJARAT': 'Gujarat',
    'HARYANA': 'Haryana',
    'HIMACHAL PRADESH': 'Himachal Pradesh',
    'JAMMU AND KASHMIR': 'Jammu & Kashmir',
    'JHARKHAND': 'Jharkhand',
    'KARNATAKA': 'Karnataka',
    'KERALA': 'Kerala',
    'MADHYA PRADESH': 'Madhya Pradesh',
    'MAHARASHTRA': 'Maharashtra',
    'MANIPUR': 'Manipur',
    'MEGHALAYA': 'Meghalaya',
    'MIZORAM': 'Mizoram',
    'NAGALAND': 'Nagaland',
    'ODISHA': 'Orissa',
    'PUNJAB': 'Punjab',
    'RAJASTHAN': 'Rajasthan',
    'SIKKIM': 'Sikkim',
    'TAMIL NADU': 'Tamil Nadu',
    'TELANGANA': 'Telangana',
    'TRIPURA': 'Tripura',
    'UTTAR PRADESH': 'Uttar Pradesh',
    'UTTARAKHAND': 'Uttaranchal',
    'WEST BENGAL': 'West Bengal',
    'ANDAMAN AND NICOBAR ISLANDS': 'Andaman & Nicobar Island',
    'CHANDIGARH': 'Chandigarh',
    'DADRA AND NAGAR HAVELI AND DAMAN AND DIU': 'Dadra & Nagar Haveli & Daman & Diu',
    'LAKSHADWEEP': 'Lakshadweep',
    'NCT OF DELHI': 'Delhi',
    'PUDUCHERRY': 'Puducherry'
}

# Apply the mapping to normalize state names
merged_df['State Name'] = merged_df['State Name'].apply(lambda x: state_name_mapping.get(x, x))

# Check and print normalized state names
print(merged_df['State Name'].unique())

if selected == "Home":
    st.image("D:\Swetha Documents\RE-CT_PROJECT\IHR.png",width=None,caption="Industrial HR",use_column_width=True)
    # Dataset
    st.subheader(":violet[ABOUT THE DATASET]")
    st.markdown("""
    - Our dataset comprises state-wise counts of main and marginal workers across diverse industries, including manufacturing, construction, retail, and more.
    - Explore the dynamic landscape of India's workforce with our Industrial Human Resource Geo-Visualization project.
    - Gain insights into employment trends, industry distributions, and economic patterns to drive informed decision-making and policy formulation.""")

    # Key Features
    st.subheader(":violet[KEY FEATURES]")
    st.markdown("""
    - **:red[Data Exploration:]** Dive deep into state-wise industrial classification data.
    - **:red[Visualization:]** Interactive charts and maps for intuitive data exploration.
    - **:red[Natural Language Processing:]** Analyze and categorize core industries using NLP techniques.
    - **:red[Insights and Analysis:]** Extract actionable insights to support policy-making and resource management.
    """)

        # Technologies Used
    st.subheader(":violet[TECHNOLOGIES USED]")
    st.write("We leverage cutting-edge technologies such as:")
    st.markdown("""
    - Python for data processing and analysis.
    - Streamlit for interactive visualization.
    - Plotly and Matplotlib for creating insightful charts.
    - NLTK for Natural Language Processing tasks.
    """)

    # About the Project
    st.subheader(":violet[ABOUT THE PROJECT]")
    st.write("Our project is about:")
    st.markdown("""
    - Update and refine the industrial classification data of main and marginal workers.
    - Provide accurate and relevant information for policy-making and employment planning.
    - Empower stakeholders with actionable insights to foster economic growth and development.
    """)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------#

elif selected == "Data Visualization":
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(merged_df['NIC Name'])

    # KMeans Clustering
    num_clusters = 5  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(X_tfidf)

    st.markdown(
    """
    <div style="text-align: center;">
        <h2 style="color: red;">INDUSTRIAL HUMAN RESOURCE DASHBOARD</h2>
    </div>
    """,
    unsafe_allow_html=True)
    # Select box for type of worker
    worker_type = st.radio(':green[Select Worker Type]',['Main Workers', 'Marginal Workers'])

    # Column mapping
    if worker_type == 'Main Workers':
        column_total = 'Main Workers - Total -  Persons'
        column_rural = 'Main Workers - Rural -  Persons'
        column_urban = 'Main Workers - Urban -  Persons'
    else:
        column_total = 'Marginal Workers - Total -  Persons'
        column_rural = 'Marginal Workers - Rural -  Persons'
        column_urban = 'Marginal Workers - Urban -  Persons'

    # Strip any extra spaces from column names
    merged_df.columns = [col.strip() for col in merged_df.columns]

    # Print DataFrame columns for debugging
    print("DataFrame Columns:", merged_df.columns)

    # Scatter Plot
    fig1 = px.scatter(merged_df, x=column_total, y=column_rural, color='Cluster', title=f'{worker_type} - Total vs Rural',color_discrete_sequence=px.colors.qualitative.Dark2_r)
    st.plotly_chart(fig1)

    fig2 = px.scatter(merged_df, x=column_total, y=column_urban, color='Cluster', title=f'{worker_type} - Total vs Urban')
    st.plotly_chart(fig2)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------#

    # Box Plot for Top 10 NIC Names
    top_10_nic_names = merged_df['NIC Name'].value_counts().head(10).index
    top_10_df = merged_df[merged_df['NIC Name'].isin(top_10_nic_names)]

    fig3 = px.box(top_10_df, x='NIC Name', y=column_total, title=f'{worker_type} by Top 10 NIC Names',color_discrete_sequence=px.colors.qualitative.Dark24_r)
    st.plotly_chart(fig3)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------#

    # Cluster Distribution
    st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: green;">DISTRIBUTION OF CLUSTERS</h1>
    </div>
    """,
    unsafe_allow_html=True)

    # Count the occurrences of each cluster
    cluster_counts = merged_df['Cluster'].value_counts()

    # Convert counts to a pie chart
    fig4, ax = plt.subplots()
    ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig4.gca().add_artist(centre_circle)
    st.pyplot(fig4)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------#
    # TF-IDF Vectorization for Word Cloud
    text = ' '.join(merged_df['NIC Name'])
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Count word frequency
    word_freq = Counter(tokens)
    top_words = word_freq.most_common(10)

    # Generate word cloud using the top words
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_words))

    # Display the word cloud in Streamlit
    st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: darkblue;">WORD CLOUD FOR TOP 10 NIC NAMES</h1>
    </div>
    """,
    unsafe_allow_html=True)
    st.image(wordcloud.to_array(), caption='Word Cloud', use_column_width=True)
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
    # Word Cloud for selected cluster
    selected_cluster = st.selectbox(':red[Select Cluster]', range(num_clusters))

    # Filter text data for the selected cluster
    text_for_cluster = merged_df[merged_df['Cluster'] == selected_cluster]['NIC Name']

    # Tokenize and clean text data
    tokens = word_tokenize(' '.join(text_for_cluster))
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]

    # Count word frequency
    word_freq = Counter(tokens)

    # Generate word cloud for the selected cluster
    wordcloud_cluster = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(word_freq))

    # Display the word cloud in Streamlit
    st.markdown(
    f"""
    <div style='text-align: center;'>
        <h1 style='color: darkviolet;'>WORD CLOUD FOR CLUSTER {selected_cluster}</h1>
    </div>
    """,
    unsafe_allow_html=True)
    st.image(wordcloud_cluster.to_array(), caption='Word Cloud', use_column_width=True)
    merged_df = merged_df.dropna(subset=['State Name'])

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
elif selected == "Geo Visualization":
    @st.cache_resource
    def fetch_geojson():
        geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
        response = requests.get(geojson_url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch GeoJSON data")

    # Main Streamlit App
    def main():
        st.markdown(
            """
            <div style="text-align: center;">
            <h1 style="color: Yellow;">INDIA MAP VISUALIZATION</h1>
            </div>
            """,unsafe_allow_html=True)

        # Fetch GeoJSON data
        geojson_data = fetch_geojson()

        # Extract state names from GeoJSON data
        geojson_state_names = set(feature['properties']['NAME_1'] for feature in geojson_data['features'])

        # State names from DataFrame
        dataframe_state_names = set(merged_df['State Name'])

        # Select box for type of worker
        worker_type = st.radio(':green[Select Worker Type]',['Main Workers', 'Marginal Workers'], key="worker_type_selectbox")

        # Select box for sex
        sex_type = st.radio(':green[Select Sex]', ['Males', 'Females'], key="sex_type_selectbox")

        # Select box for area
        area_type = st.radio(':green[Select Area]', ['Rural', 'Urban'], key="area_type_selectbox")

        # Determine the column based on selected worker type, sex, and area
        column_name = f'{worker_type} - {area_type} - {sex_type}'

        # Plotly Choropleth map
        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=merged_df['State Name'],  # Use the column with state names
            featureidkey="properties.NAME_1",  # Key in geojson to match with DataFrame
            z=merged_df[column_name],  # Use the column for analysis
            colorscale='inferno',
            zmin=merged_df[column_name].min(),
            zmax=merged_df[column_name].max(),
            marker_opacity=0.7,
            marker_line_width=0,
        ))

        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_zoom=3,
            mapbox_center={"lat": 20.5937, "lon": 78.9629},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            title=f"{worker_type} ({sex_type}, {area_type}) Population Across Indian States",
            title_x=0.5
        )
        # Display the map
        st.plotly_chart(fig)

    # Call the main function
    if __name__ == "__main__":
        main()

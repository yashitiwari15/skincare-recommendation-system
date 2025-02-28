import streamlit as st
from streamlit_option_menu import option_menu
from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Import the Dataset 
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="Skin Care Recommender System", page_icon=":rose:", layout="wide",)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2

def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Skin Care", "Get Recommendation", "Skin Care 101"],  # required
                icons=["house", "stars", "book"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Skin Care":
    st.title(f"{selected} Product Recommender :sparkles:")
    st.write('---') 

    st.write(
        """
        ##### **This Skin Care Product Recommendation Application is a Machine Learning implementation that provides skin care product recommendations tailored to your skin type and concerns.**
        """)
    
    # Displaying a local video file
    video_file = open("skincare.mp4", "rb").read()
    st.video(video_file, start_time=1)  # Displaying the video 
    
    st.write(' ') 
    st.write(' ')
    st.write(
        """
        ##### You will get skin care product recommendations from various cosmetic brands with over 1200+ products tailored to your skin needs. 
        ##### There are 5 categories of skin care products for 5 different skin types, as well as concerns and benefits you want from the products. This recommendation app is just a system that provides suggestions based on the data you enter, not a scientific consultation.
        ##### Please select the *Get Recommendation* page to start receiving recommendations or choose the *Skin Care 101* page to see tips and tricks about skin care.
        """)
    
    st.write(
        """
        **Enjoy :) !**
        """)
    
    
    st.info('Credit: Created by Dwi Ayu Nouvalina')

if selected == "Get Recommendation":
    st.title(f"Let's {selected}")
    
    st.write(
        """
        ##### **To get recommendations, please enter your skin type, concerns, and desired benefits to get suitable skin care product recommendations.**
        """) 
    
    st.write('---') 

    first,last = st.columns(2)

    # Choose a product type category
    category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique())
    category_pt = skincare[skincare['product_type'] == category]

    # Choose a skin type
    skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'])
    category_st_pt = category_pt[category_pt[skin_type] == 1]

    # Select concerns
    prob = st.multiselect(label='Skin Problems : ', options= ['Dull Skin', 'Acne', 'Acne Scars', 'Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin'])

    # Choose notable effects
    options_ne = category_st_pt['notable_effects'].unique().tolist()
    selected_options = st.multiselect('Notable Effects : ', options_ne)
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    # Choose product
    options_pn = category_ne_st_pt['product_name'].unique().tolist()
    product = st.selectbox(label='Recommended Products for You:', options=sorted(options_pn))

    ## MODELLING with Content Based Filtering
    tf = TfidfVectorizer()

    tf.fit(skincare['notable_effects']) 
    tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 
    cosine_sim = cosine_similarity(tfidf_matrix) 
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    def skincare_recommendations(product_name, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description']], k=5):
        index = similarity_data.loc[:,product_name].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1:-(k+2):-1]]
        closest = closest.drop(product_name, errors='ignore')
        df = pd.DataFrame(closest).merge(items).head(k)
        return df

    model_run = st.button('Find More Product Recommendations!')
    if model_run:
        st.write('Here are other similar products recommended for you:')
        st.write(skincare_recommendations(product))

if selected == "Skin Care 101":
    st.title(f"Take a Look at {selected}")
    st.write('---') 

    st.write(
        """
        ##### **Below are tips and tricks you can follow to maximize the use of your skin care products.**
        """) 
    
    image = Image.open('imagepic.jpg')
    st.image(image, caption='Skin Care 101')
    

    st.write(
        """
        ### **1. Facial Wash**
        - Use facial wash products recommended or suitable for you.
        - Wash your face a maximum of twice a day (morning and night). Over-washing can strip natural oils.
        - Avoid scrubbing your face harshly, as it can damage the skin barrier.
        - The best way to clean your skin is to use your fingertips in circular motions for 30-60 seconds.
        """)

    st.write(
        """
        ### **2. Toner**
        - Use a toner that is recommended or suits your skin type.
        - Apply toner with a cotton pad, then pat gently with your hands to absorb it better.
        - Use toner after washing your face.
        - For sensitive skin, avoid products with fragrance.
        """)
    
    # Further skin care tips...
    # (Continued for other tips like serum, moisturizer, sunscreen)

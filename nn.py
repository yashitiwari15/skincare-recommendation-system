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
        ##### **This Skin Care Product Recommendation App implements Machine Learning to recommend skin care products based on your skin type and concerns.**
        """)
    
    # Displaying a local video file
    video_file = open("skincare.mp4", "rb").read()
    st.video(video_file, start_time = 1) # Displaying the video 
    
    st.write(' ') 
    st.write(' ')
    st.write(
        """
        ##### You will receive skin care product recommendations from various brands with over 1200+ products tailored to your skin needs. 
        ##### There are 5 categories of skin care products for 5 different skin types, as well as concerns and benefits you wish to achieve from the product. This recommendation app is purely a system that provides suggestions based on the data you input, not scientific consultation.
        ##### Please select the *Get Recommendation* page to start receiving recommendations or the *Skin Care 101* page for skin care tips and tricks.
        """)
    
    st.write(
        """
        **Happy Trying :) !**
        """)
    
    
    st.info('Credit: Created by Yashi Tiwari , Meenal Khatri , Anshika Jain')

if selected == "Get Recommendation":
    st.title(f"Let's {selected}")
    
    st.write(
        """
        ##### **To get recommendations, please input your skin type, concerns, and desired benefits to receive suitable product suggestions.**
        """) 
    
    st.write('---') 

    first,last = st.columns(2)

    # Choose a product product type category
    # pt = product type
    category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique() )
    category_pt = skincare[skincare['product_type'] == category]

    # Choose a skin type
    # st = skin type
    skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'] )
    category_st_pt = category_pt[category_pt[skin_type] == 1]

    # Select concerns
    prob = st.multiselect(label='Skin Problems : ', options= ['Dull Skin', 'Acne', 'Acne Scars','Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin'] )

    # Choose notable_effects
    # From the filtered products based on product type and skin type, extract unique notable_effects
    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    selected_options = st.multiselect('Notable Effects : ',opsi_ne)
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    # Choose product
    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    product = st.selectbox(label='Recommended Products for You:', options = sorted(opsi_pn))

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
        ##### **Here are tips and tricks to maximize the use of skin care products.**
        """) 
    
    image = Image.open('imagepic.jpg')
    st.image(image, caption='Skin Care 101')
    
    # The tips are listed in English
    st.write(
        """
        ### **1. Facial Wash**
        - Use facial wash products that are recommended or suitable for you.
        - Wash your face no more than twice a day (morning and night). Washing too often strips the skin's natural oils.
        - Avoid scrubbing your face too harshly.
        - Cleanse gently with circular motions using your fingertips for 30-60 seconds.
        """)
    
    st.write(
        """
        ### **2. Toner**
        - Use a recommended or suitable toner.
        - Apply with a cotton pad, then pat gently with your hands for better absorption.
        - Use toner after cleansing your face.
        - If you have sensitive skin, avoid toners with fragrance.
        """)
    
    st.write(
        """
        ### **3. Serum**
        - Use serums recommended or suitable for your needs.
        - Apply after cleansing for optimal absorption.
        - Choose serums tailored to your skin concerns (acne scars, anti-aging, etc.).
        """)
    
    st.write(
        """
        ### **4. Moisturizer**
        - A must-have for locking in hydration and nutrients.
        - Use different moisturizers for day and night.
        - Allow 2-3 minutes for the serum to absorb before applying moisturizer.
        """)
    
    st.write(
        """
        ### **5. Sunscreen**
        - Use sunscreen to protect against UV rays.
        - Reapply every 2-3 hours.
        - Wear sunscreen indoors as UV rays penetrate windows.
        """)
    
    st.write(
        """
        ### **6. Stay Consistent**
        - Regular and consistent care is key to visible results.
        """)

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import  Image
import random

# list of all 101 class names
class_names = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheesecake',
 'cheese_plate',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']


# loading the modle
@st.cache(allow_output_mutation=True) # setting up cache for the model
def load_model():
    model = tf.keras.models.load_model('effi_080_second.h5')
    return model

# call the model to predict the class of the image
model = load_model()

# showing a Header
# st.title('Food 101 Classifier™')
st.markdown("<h1 style='text-align: center;'>Food 101 Classifier™</h1>", unsafe_allow_html=True)
st.write('A image classifier based on the Food 101 dataset')
col1, col2 = st.beta_columns(2)


# Asking for file
file = col2.file_uploader("Upload an image of food", type=["png", "jpg"])
#food images list
sam_lst = ['None', 'Icecream', 'Pizza', 'Waffels', 'Steak']
# random greet !!!
g_lst = ['you ordered  >>  ', 'so you like  >>  ', 'want to have some  >> ', "your today's lunch >>  ", "your favorite food is >>  ", "serving  >>  "]
# getting random greeting
greet =  random.choice(g_lst)


# function for predicting food class with a custom image
def predict_class(file, greet):
    """Functon that will prepare the images and will predict the class"""
    img = Image.open(file)
    img2 = img.copy()
    img2.resize((300, 300))
    col1.image(img2,caption=f"Looks Delicious!! ", use_column_width=True)
    # converting the image to a numpy array
    img_array = np.array(img)
    # reshaping the image to a 4d tensor usable by the model
    img = tf.image.resize(img_array, size=(224,224))
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred_cls = class_names[pred.argmax()]
    col2.success(greet + pred_cls)  # showing the prediction class name


# prdeicting the class of the image from the file / custome image and samples
if file is not None:
    with st.spinner('Hold on your food is getting cooked...'):
         predict_class(file, greet)
else:
    col1.warning("No image uploaded. You can use sample imgaes from below list")
    file2 = col1.selectbox('Select from sample images', options=sam_lst)
    if file2 == 'Icecream':
        file = 'samples\icecream.jpg'
        with st.spinner('Hold on your food is getting cooked...'):
            predict_class(file, greet)
    elif file2 == 'Pizza':
        file = 'samples\pizza.jpg'
        with st.spinner('Hold on your food is getting cooked...'):
            predict_class(file, greet)
    elif file2 == 'Waffels': 
        file = 'samples\waffels.jpg'
        with st.spinner('Hold on your food is getting cooked...'):
            predict_class(file, greet)    
    elif file2 == 'Steak':
        file = 'samples\steak.jpg'
        with st.spinner('Hold on your food is getting cooked...'):    
            predict_class(file, greet) 
    else:
        pass
    
note = """ 
\n
This project based on the [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) Paper which used Convolutional Neuranetwork trained for 2 to 3 days to achieve 77.4% top-1 accuracy.
The project is made by download the food101 dataset from the [TensorFlow dataset](https://www.tensorflow.org/datasets/catalog/food101)(size: 4.6GB) which consists of 750 images x 101 training classes = 75750 training images.
I used the [EfficientNetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0) model with fine-tune unfreeze all layers of the model. \n
Although this WebApp model accuracy is around 80% to 82%. I am also sharing the [notebook](https://colab.research.google.com/drive/15sJJhrZBo12CA3flnrX-NC4WwrP84z0D?usp=sharing) for this project.
[Github](https://github.com/subha996/food-101_updated)
"""
st.write(note)







with st.beta_expander('Food Names(Classes), The model will work better if you chose food from this list'):
 st.write(class_names)
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import time
# Custom HTML/CSS for the banner
# custom_html = """
# <div class="banner">
#     <img src="https://brand.charlotte.edu/wp-content/uploads/sites/183/2023/04/8716-01-Charlotte-Master-File-v7_1.png" alt="Banner Image">
# </div>
# <style>
#     .banner {
#         width: 35%;
#         height: 600px;
#         overflow: visible;
#     }
#     .banner img {
#         width: 100%;
#         object-fit: fill;
#     }
# </style>
# """
# # Display the custom HTML
# st.components.v1.html(custom_html)

# st.set_page_config(layout="wide")
st.image("logo.png")

# Sidebar content
# st.sidebar.header("Sidebar Title")
# st.sidebar.subheader("Subheading")
# st.sidebar.text("Produced by UNCC")

# Main content
st.title("Welcome to the Ultrasonic Testing Data Synthesizer!")
new_title3 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 28px;">Dear Professor Chakra and Bharathi</p>'
st.markdown(new_title3, unsafe_allow_html=True)
st.write("The bulk of our project is contained within the Pre-Trained Synthesizers tab, which showcases work of 42 various VAE models that have been trained. The VAE model framework has been optimized to a temporarily satisfactory point (however further optimization could still be implemented, as there stands to be room for improvement). In the streamlit application that has been submitted, only 1 of the 42 models are shown, and the other 41 are still being input into the application (Placeholders are currently shown for the other 41 models). Additionally, experiments were performed with an alternative model framework (GAN), of which will be input into streamlit before the end of the project as well. A few odds and ends of the streamlit application still need to be touched up (ie visual aspects such as creating a more extensive home page). Lastly, the Custom Synthesizer Tab is an additional feature within our streamlit app that allows the user to train their own models based on their specfic needs (as the scope of this project simply focuses on ONE material type of ONE component thickness), however as previously stated this portion of the app is simply an additional feature to the bulk of the project.")

new_title = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 28px;">Why Synthesize Ulltrasonic Testing (UT) Data?</p>'
st.markdown(new_title, unsafe_allow_html=True)
st.write("UT data is not easy to come by. There are primarily only 2 ways to acquire UT data; to collect the data on a physical component with an encoding setup, or to acquire data via simulation softwares. Collecting data on a physical sample can be time, cost, and labor intensive. Using simulation sotwares to acquire UT data is time and cost intensive, but the data that is produced by simulation softwares often certain lack realistic features such as material and weld interface noise, coupling induced signal effects, ect.")
new_title2 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 28px;">Capability of the UT Data Synthesizer</p>'
st.markdown(new_title2, unsafe_allow_html=True)
st.write("This application allows you to synthesize any type of UT amplitude scans that you desire a larger quantity of unique data for, along a number of post processing and visualiztion tools in this application that can be used to exam the synthetic data's level of realism. There are also pre-built synthesizers in this application that are able to produce amplitude scans (A-scans) of corrosion and backwall of an aluminum component .75 inches (19.05mm) in thickness. This application supplies over 40 models that can synthesize corrosion of through-wall ranging from 66% to 2%, as well uncorroded material.")
st.sidebar.image('logo_green.png', use_column_width=True)










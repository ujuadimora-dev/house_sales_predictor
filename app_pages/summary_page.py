import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def summary_page_body():

    image_main = plt.imread(f"media/house.jpg")
    # image_isu = plt.imread(f"media/iowasu.jpg")

    st.image(image_main, caption='Housing - Ames, Iowa.')

    st.write("### Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Purpose and Motivation**\n\n"
        f" The general pupose of this project is to provide a tool that allows"
        f" a client to predict the potential sale price of a property in"
        f" Ames, Iowa, by providing detailed and typical information on the"
        f" real estate in question.\n\n"
        f" Specifically, a client has requested this app in order"
        f" to estimate the sale price for several inherited properties in "
        f" Ames, Iowa. The client has provided a publically available data set"
        f" which is used to train the machine learning model and "
        f" predict local real estate sale prices. \n \n"
        f"**Project Dataset**\n"
        f"* The data set can be accessed at "
        f"[Kaggle](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data)"  # noqa
        f" where it is hosted by Code Institute.\n"
        f"* The dataset represents a record of approx. 1500 real estate "
        f" sales in Ames, Iowa. Each record contains 23 features indicating"
        f" the house profile, such as Floor Area, Basement, Garage, "
        f" Kitchen, Lot,"
        f" Porch, Wood Deck, and Year Built. It also contains the Sale Price."
        f" The features are extensive, so please visit the site for more"
        f" information.")

    # copied from README file - "Business Requirements" section
    st.success(
        f"**Business Requirements**\n\n"
        f"The project has 3 business requirements:\n"
        f"* 1. The client is interested in understanding the correlation "
        f" between a properties attributes/features and the sale price."
        f" Therefore, the client expects data visualization of the correlated"
        f" variables against the sale prices for illustration. \n"
        f"* 2. The client is interested in predicting the potential sale "
        f" prices"
        f" for properties in Ames, Iowa, and specifically, she wants to"
        f" determine a potential value for the properties she inherited. \n"
        f"* 3. The client would like to have Dachbord for easy access."
        f" an online application."
    )

    # Link to README file, so the users can have access to full
    # project documentation
    st.write(
        f"* For additional information on this project please consult the "
        f"[README](https://github.com/ujuadimora-dev/house_sales_predictor/blob/main/README.md)"
        f" file for this project hosted on GitHub.\n"
        f"* The project was developed by Adimora Uju Louisa. To find out"
        f" more information about the developer, please visit "
        f" [LinkedIn](https://www.linkedin.com/in/adimora-uju-louisa-48b821179/)"
        f" or [GitHub](https://github.com/ujuadimora-dev). \n"
        f"* For additional information on Ames, Iowa, home of Iowa State"
        f" University and the Iowa State Cyclones, visit "
        f"[Wikipedia](https://en.wikipedia.org/wiki/Ames,_Iowa#History).")

  
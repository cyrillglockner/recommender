# Network Connector App

## Overview

The Network Connector App is a versatile platform designed to foster connections and collaborations across various industries and interest groups. Users can upload their own data, enabling the app to tailor recommendations to match individuals based on shared interests, expertise, and collaboration opportunities.

## Features

- **Custom Data Upload**: Allows users to input their dataset for personalized networking recommendations.
- **AI-Powered Matching**: Utilizes advanced algorithms to suggest highly relevant potential connections.
- **Interactive User Interface**: Easy-to-navigate platform for exploring profiles, recommendations, and making connections.

## Files in the Repository

- `.gitignore`: Specifies files and directories Git should ignore.
- `README.md`: Offers an overview, setup instructions, and additional information about the project.
- `app.py`: The main script for the app's user interface, built with Streamlit, including file upload and recommendation features.
- `fictional_users.csv`: Example dataset demonstrating the expected format for user data.
- `recommender.py`: Commandline script for the app.
- `requirements.txt`: Lists all dependencies needed to run the app.

## Installation and Setup

'streamlit run app.py'

or

'python recommender.py fictional_users.csv user_id' user_id is any id from the csv file. 

Visit http://localhost:8501 in your browser to use the app.

License
This project is licensed under the MIT License - see the LICENSE file in this repository for more details.
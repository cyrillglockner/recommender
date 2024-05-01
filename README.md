# Recommender App

## Overview

The Recommender App is a versatile tool designed to create connections between entities for many different use-cases. In this case we have a fictional database of members of a climate action community that look for valuable connections within the network. 

Users can upload their own data, enabling the app to deliver recommendations to match individuals based on shared interests, expertise, and collaboration opportunities.

## Features

- **Custom Data Upload**: Allows users to input their dataset for personalized networking recommendations.
- **AI-Powered Matching**: Utilizes advanced algorithms to suggest highly relevant potential connections.
- **Interactive User Interface**: Upload your own file and enter a user id you would like to receive recommendations for.

## Files in the Repository

- `.gitignore`: Specifies files and directories Git should ignore.
- `README.md`: Offers an overview, setup instructions, and additional information about the project.
- `app.py`: The main script for the app's user interface, built with Streamlit, including file upload and recommendation features.
- `fictional_users.csv`: Example dataset demonstrating the expected format for user data.
- `recommender.py`: Commandline script for the app.
- `requirements.txt`: Lists all dependencies needed to run the app.
- `LICENSE`: MIT License.

## Installation and Setup
Use your OpenAI API Key and export it to 'OPENAI_API_KEY' when running the app/recommender locally or set it in Streanlit when deploying the app.

'streamlit run app.py' Visit http://localhost:8501 in your browser to use the app.

or

'python recommender.py fictional_users.csv user_id' user_id is any id from the csv file. 

License
This project is licensed under the MIT License - see the LICENSE file in this repository for more details.

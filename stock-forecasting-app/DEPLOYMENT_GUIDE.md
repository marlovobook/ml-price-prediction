# DEPLOYMENT_GUIDE.md

# Deployment Guide for Stock Forecasting Application

This guide provides instructions for deploying the Stock Forecasting Application, including environment setup, configuration, and running the application in a production environment.

## Prerequisites

Before deploying the application, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)
- Git (optional, for version control)

## Environment Setup

1. **Clone the Repository**

   Clone the project repository to your local machine or server:

   ```
   git clone <repository-url>
   cd stock-forecasting-app
   ```

2. **Create a Virtual Environment**

   It is recommended to create a virtual environment to manage dependencies:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Install the required packages using pip:

   ```
   pip install -r requirements.txt
   ```

## Configuration

1. **Edit Configuration File**

   Modify the `config/config.yaml` file to set your desired model parameters and data sources. Ensure that the symbols and date ranges are appropriate for your use case.

## Running the Application

1. **Start the Streamlit Application**

   Run the Streamlit application using the following command:

   ```
   streamlit run streamlit_app/app.py
   ```

2. **Access the Application**

   Open your web browser and navigate to `http://localhost:8501` to access the Stock Forecasting Application.

## Additional Notes

- Ensure that your data sources are accessible and that any necessary API keys or credentials are configured in the `config.yaml` file.
- For production deployment, consider using a web server like Nginx or Apache to serve the Streamlit application.
- Monitor the application logs for any errors or issues during runtime.

## Troubleshooting

- If you encounter issues, check the console output for error messages.
- Ensure all dependencies are correctly installed and compatible with your Python version.
- Refer to the README.md for additional documentation and usage instructions.

## Conclusion

You are now ready to deploy the Stock Forecasting Application. For further assistance, please refer to the project's documentation or seek help from the community.
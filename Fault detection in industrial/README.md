The archive contains a web application for detecting faults in time series data based on PCA and DPCA algorithms,
T2 and Q statistics.

File description:

Report.pdf - Report file for this project;  
Fault_detection.ipynb - Draft for all files in directory;\
Draft_for_Fault_detection.pdf  - PDF format for the file Fault_detection.ipynb;  
fault_detector.py - File which contain realization for decomposition and statistics (main calculation file);  
app.py - File which contain web application for user interaction with algorithm in fault_detector.py;  
Streamlit_app.pdf - The application appearance. First page - sidebar, second page - main body;  
data - The folder in which the studied datasets should be located. Files should be in form .mat;  

In order to run the application, the computer must be installed python, libraries: streamlit, sklearn, numpy and pandas.
To run the application in the command line, in the directory with the app.py file, enter "streamlit run app.py". After a while, the application will open in
the browser. If this does not happen, then a link to the URL will appear in the command line to open in the search bar.

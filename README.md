# Test-Canary: Miscellaneous Files for ML + Web Development
This project is a collection of miscellaneous files used for machine learning and web development. It includes a variety of scripts and configuration files that demonstrate different aspects of these fields, such as training a convolutional neural network, creating a simple web page, and performing basic arithmetic operations. The project was created with the aim of providing a comprehensive set of tools and resources for both machine learning and web development enthusiasts. It is designed with simplicity and ease of use in mind, making it suitable for beginners and experts alike. The project's purpose is to simplify the process of learning and applying machine learning and web development concepts by providing ready-to-use scripts and configuration files. It achieves this by providing a variety of scripts that demonstrate different aspects of machine learning and web development, allowing users to learn by doing and to customize the scripts to suit their specific needs.

## Installation and Usage
To use this project, you will need Python and PyTorch installed on your machine. Here are the step-by-step instructions to install and use the project:

1. Install Python on your machine. You can download Python from the official website. Make sure to install the latest version.
2. Install PyTorch on your machine. You can install PyTorch by following the instructions on the official PyTorch website.
3. Clone the repository to your local machine using the command `git clone https://github.com/lukejagg/test-canary.git`.
4. Navigate to the project directory using the command `cd test-canary`.
5. Run the scripts in your local environment. For example, to run the training script, use the command `python train.py`. Make sure to navigate to the directory containing the script before running the command.

For detailed instructions on how to use individual scripts, please refer to the respective files.

## Dependencies
This project requires the following software and libraries:
* Python
* PyTorch

## File Descriptions
* `config.yaml`: This file contains the configuration for the machine learning model, including the model name, number of classes, and training parameters. It is used to set up and customize the model for training. The configuration parameters can be easily modified to suit different machine learning models and training requirements. Users can modify this file to customize the machine learning model according to their needs.
* `frontend.css`: This file contains the CSS styles for the web page. It defines the look and feel of the web page. The styles are written in a modular and scalable manner, making it easy to modify and extend. Users can modify this file to customize the appearance of the web page.
* `train.py`: This file contains the script for training the convolutional neural network. It uses the configuration from `config.yaml` to train the model. The script includes functions for loading data, building the model, training the model, and saving the trained model. Users can modify this file to customize the training process.
* `calculator.py`: This file contains a simple calculator program. It demonstrates basic arithmetic operations in Python. The calculator supports operations like addition, subtraction, multiplication, and division. Users can use this program to perform basic arithmetic operations.

## Usage Examples
To run the training script, use the following command:
```
python train.py
```
To use the calculator program, use the following command:
```
python calculator.py
```

## Known Issues and Limitations
Currently, there are no known issues or limitations. However, please note that the machine learning model used in this project is a basic convolutional neural network and may not provide the best performance for complex datasets. Also, the web page created using the provided CSS styles is a simple page and may not include advanced features like responsiveness or animations. Please report any issues you encounter by creating a new issue on the GitHub repository.

## Troubleshooting
If you encounter any problems while using the project, please refer to the following solutions:

* Error: 'Python is not recognized as an internal or external command, operable program or batch file.' - This error occurs when Python is not installed on your machine or the Python installation directory is not added to the PATH environment variable. To fix this, install Python and add the Python installation directory to the PATH environment variable.
* Error: 'No module named torch' - This error occurs when PyTorch is not installed on your machine. To fix this, install PyTorch by following the instructions on the official PyTorch website.

## Future Plans
The project is continuously being improved and new features are being added. Some of the planned features include:

* Adding more machine learning models to the project.
* Improving the web page by adding more advanced features like responsiveness and animations.
* Adding more scripts that demonstrate different aspects of machine learning and web development.

## Contributing
Contributions are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch in your forked repository.
3. Make your changes in the new branch.
4. Submit a pull request from the new branch to the main branch of the original repository.

For major changes, please open an issue first to discuss what you would like to change. When submitting a pull request, please make sure to include a detailed description of the changes you made and the issue it addresses. Also, please follow the coding standards and conventions used in the project.

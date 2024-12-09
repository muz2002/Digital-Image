# Application Name

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Contribution](#contribution)
- [License](#license)

## Features

- **Image Upload**: Easily upload images through a user-friendly interface.
- **Model Selection**: Choose from multiple pre-trained CNN models via the sidebar.
- **Instant Predictions**: Receive immediate predictions or detections after uploading an image.
- **Interactive Carousel**: A carousel on the homepage guides users on how the application works.

## Project Structure

```.
├── config
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── core
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── views.py
│   ├── templates
│   │   └── pages
│   │       └── index.html
│   └── static
│       └── assets
│           ├── img
│           └── scss
│               └── black-dashboard
│                   └── bootstrap
│                       └── _custom-forms.scss
├── media
│   └── uploaded.png
├── manage.py
└── requirements.txt
```

## Prerequisites

- Python 3.x
- Django
- Pillow (for image processing)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd your-repo-name
   ```

3. **Create a virtual environment and activate it:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Apply migrations:**

   ```bash
   python manage.py migrate
   ```

2. **Start the development server:**

   ```bash
   python manage.py runserver
   ```

3. **Access the application:**
   Open your browser and navigate to [http://localhost:8000/](http://localhost:8000/).

## How It Works

1. **Upload an Image:**
   On the homepage, click the upload button and select an image from your device.

2. **Select a CNN Model:**
   Choose a pre-trained CNN model from the sidebar options.

3. **Get Instant Results:**
   Submit the form to receive immediate predictions or detections displayed on the page.

## Screenshots

![choose](https://github.com/user-attachments/assets/e494eacc-3a19-413d-972e-43056ea57192)

![choose2](https://github.com/user-attachments/assets/1b31b6f8-a02c-41f1-9e8b-e38fd59d8595)

![choose3](https://github.com/user-attachments/assets/0bee20dd-bbee-450d-ab31-410aceff5163)

## Contribution

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License

This project is open-source and available under the [MIT License](LICENSE).

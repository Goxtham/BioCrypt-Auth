BioCrypt-Auth

A blockchain-based authentication system that combines facial recognition, eye biometrics, and password verification for enhanced security.

Features

Multi-factor authentication using:

Face biometrics (DeepFace-based similarity matching)

Eye biometrics

Password verification

Blockchain-based data storage for enhanced security

AES encryption for sensitive data

SHA-256 hashing for data integrity

SQLite database for user management

Requirements

bash
pip install -r requirements.txt

Installation

Clone the repository
bash
git clone https://github.com/yourusername/BioCrypt-Auth.git
cd BioCrypt-Auth

Install dependencies
bash:README.md
pip install pillow numpy pycryptodome opencv-python deepface

Initialize the database
bash
python app.py

Usage

Register a new user:
python
auth_system = HybridCryptoAuth()
success = auth_system.register_user(
user_id="unique_user_id",
eye_image_path="path/to/eye/image.jpg",
face_image_path="path/to/face/image.jpg",
password="secure_password123"

Verify a user:
python
is_verified, message = auth_system.verify_user(
user_id="unique_user_id",
eye_image_path="path/to/eye/image.jpg",
face_image_path="path/to/face/image.jpg",
password="secure_password123"
)

System Architecture

Input Processing

Image preprocessing and binary conversion

Password to binary conversion

Feature extraction using DeepFace

Automatic deletion of temporary login images after authentication

Security Measures

AES encryption for pattern storage

SHA-256 hashing for data integrity

Blockchain implementation for tamper protection

Storage

SQLite database for user data

Blockchain for encrypted patterns

Secure hash linking between blocks

Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
## License

[MIT](https://choosealicense.com/licenses/mit/)



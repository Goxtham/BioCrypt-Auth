import hashlib
import os
from PIL import Image
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import cv2
import json
from datetime import datetime
import sqlite3

class HybridCryptoAuth:
    def __init__(self, db_path='auth_system.db'):
        self.key = get_random_bytes(32)  # AES-256 key
        self.blockchain = []
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                encrypted_pattern BLOB,
                block_hash TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def preprocess_image(self, image_path):
        """Convert image to binary features"""
        try:
            # For AVIF format, use pillow-avif plugin
            if image_path.lower().endswith('.avif'):
                from pillow_avif import AvifImagePlugin
                AvifImagePlugin.register()  # Register AVIF format support
            
            # For WEBP format, ensure Pillow has WEBP support
            elif image_path.lower().endswith('.webp'):
                if 'WEBP' not in Image.OPEN:
                    raise ValueError("WEBP support not available. Please install webp support for Pillow")
            
            # Handle different image formats using PIL
            pil_image = Image.open(image_path)
            
            # Convert to RGB mode if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL image to numpy array
            img_array = np.array(pil_image)
            
            # Convert RGB to BGR (OpenCV format)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img = cv2.resize(img, (256, 256))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            binary_string = ''.join(['1' if pixel == 255 else '0' for pixel in binary.flatten()])
            
            return binary_string
        
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def password_to_binary(self, password):
        """Convert password to binary string"""
        password_bytes = password.encode('utf-8')
        binary = bin(int.from_bytes(password_bytes, byteorder='big'))[2:]
        return binary.zfill(len(password_bytes) * 8)

    def merge_features(self, eye_binary, face_binary, password_binary):
        """Merge binary features using a simple interleaving pattern"""
        max_length = max(len(eye_binary), len(face_binary), len(password_binary))
        eye_binary = eye_binary.ljust(max_length, '0')
        face_binary = face_binary.ljust(max_length, '0')
        password_binary = password_binary.ljust(max_length, '0')

        merged = ''
        for i in range(max_length):
            merged += eye_binary[i] + face_binary[i] + password_binary[i]
        
        return merged

    def apply_sha256(self, data):
        """Apply SHA-256 hashing"""
        return hashlib.sha256(data.encode()).hexdigest()

    def shift_aes_encrypt(self, data):
        """Encrypt data using AES"""
        cipher = AES.new(self.key, AES.MODE_CBC)
        padded_data = data + (16 - len(data) % 16) * chr(16 - len(data) % 16)
        encrypted_data = cipher.encrypt(padded_data.encode())
        return cipher.iv + encrypted_data

    def shift_aes_decrypt(self, encrypted_data):
        """Decrypt data using AES"""
        iv = encrypted_data[:16]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        decrypted_data = cipher.decrypt(encrypted_data[16:])
        padding_length = decrypted_data[-1]
        return decrypted_data[:-padding_length].decode()

    def add_to_blockchain(self, user_id, encrypted_pattern):
        """Add encrypted pattern to blockchain"""
        block = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'encrypted_pattern': encrypted_pattern.hex(),
            'previous_hash': self.blockchain[-1]['hash'] if self.blockchain else '0' * 64
        }
        
        block_string = json.dumps(block, sort_keys=True)
        block['hash'] = hashlib.sha256(block_string.encode()).hexdigest()
        
        self.blockchain.append(block)
        return block

    def save_user_to_db(self, user_id, encrypted_pattern, block_hash):
        """Save user data to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (user_id, encrypted_pattern, block_hash)
            VALUES (?, ?, ?)
        ''', (user_id, encrypted_pattern, block_hash))
        conn.commit()
        conn.close()

    def get_user_from_db(self, user_id):
        """Retrieve user data from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT encrypted_pattern, block_hash FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result

    def user_exists(self, user_id):
        """Check if a user already exists in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users WHERE user_id = ?', (user_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0

    def register_user(self, user_id, eye_image_path, face_image_path, password):
        """Register a new user"""
        try:
            # Check if user already exists
            if self.user_exists(user_id):
                raise Exception(f"User ID '{user_id}' already exists. Please choose a different ID.")

            eye_binary = self.preprocess_image(eye_image_path)
            face_binary = self.preprocess_image(face_image_path)
            password_binary = self.password_to_binary(password)

            merged_pattern = self.merge_features(eye_binary, face_binary, password_binary)
            hashed_pattern = self.apply_sha256(merged_pattern)
            encrypted_pattern = self.shift_aes_encrypt(hashed_pattern)

            block = self.add_to_blockchain(user_id, encrypted_pattern)
            self.save_user_to_db(user_id, encrypted_pattern, block['hash'])

            return True

        except Exception as e:
            raise Exception(f"Registration failed: {str(e)}")

    def verify_user(self, user_id, eye_image_path, face_image_path, password):
        """Verify user authentication"""
        try:
            user_data = self.get_user_from_db(user_id)
            if not user_data:
                return False, "User not found in the database."

            stored_encrypted_pattern, stored_block_hash = user_data

            eye_binary = self.preprocess_image(eye_image_path)
            face_binary = self.preprocess_image(face_image_path)
            password_binary = self.password_to_binary(password)

            current_pattern = self.merge_features(eye_binary, face_binary, password_binary)
            current_hash = self.apply_sha256(current_pattern)

            stored_hash = self.shift_aes_decrypt(stored_encrypted_pattern)

            if current_hash == stored_hash:
                return True, "Verification successful!"
            else:
                return False, "Verification failed: Data does not match."

        except Exception as e:
            return False, f"Verification failed: {str(e)}"

# Example usage
if __name__ == "__main__":
    auth_system = HybridCryptoAuth()
    
    # Generate a unique user ID using timestamp
    user_id = f"user_{int(datetime.now().timestamp())}"
    
    # Paths for registration
    eye_image_path_register = "C:/Users/goxth/Documents/COLLEGE WORKS/FINAL YEAR PROJECT/Eye_iris.jpg"
    face_image_path_register = "C:/Users/goxth/Documents/COLLEGE WORKS/FINAL YEAR PROJECT/pexels-simon-robben-55958-614810.jpg"
    password_register = "secure_password123"

    # Paths for verification (initially same as registration for testing successful verification)
    eye_image_path_verify = r'C:\Users\goxth\Documents\COLLEGE WORKS\FINAL YEAR PROJECT\istockphoto-179209374-612x612.jpg' # Can be changed to test different eye images
    face_image_path_verify = r'C:\Users\goxth\Documents\COLLEGE WORKS\FINAL YEAR PROJECT\pexels-photo-2379005.jpeg'
    password_verify = "secure_password123"  # Can be changed to test different passwords

    print("\n=== Registration Phase ===")
    print(f"Registering with:")
    print(f"Eye Image: {eye_image_path_register}")
    print(f"Face Image: {face_image_path_register}")
    print(f"Password: {password_register}")

    # Example registration
    try:
        success = auth_system.register_user(
            user_id=user_id,
            eye_image_path=eye_image_path_register,
            face_image_path=face_image_path_register,
            password=password_register
        )
        print(f"\nRegistration successful for user {user_id}!" if success else "Registration failed!")
        
        print("\n=== Verification Phase ===")
        print(f"Verifying with:")
        print(f"Eye Image: {eye_image_path_verify}")
        print(f"Face Image: {face_image_path_verify}")
        print(f"Password: {password_verify}")

        # Example verification
        is_verified, message = auth_system.verify_user(
            user_id=user_id,
            eye_image_path=eye_image_path_verify,
            face_image_path=face_image_path_verify,
            password=password_verify
        )
        print(f"\nVerification Result: {message}")
        
    except Exception as e:
        print(f"Error: {str(e)}")